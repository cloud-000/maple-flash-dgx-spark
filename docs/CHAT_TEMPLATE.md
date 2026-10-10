# Chat Template & Reasoning Retention

> **Document status:** Design specification and technical reference for Maple's chat template, role conventions, and reasoning (`<think>`) trace retention policies across single-turn and multi-turn conversations.

Maple uses a ChatML-based template defined in [`checkpoints/maple-2bit/chat_template.jinja`](../checkpoints/maple-2bit/chat_template.jinja), validated by the test suite in [`tests/test_chat_template.py`](../tests/test_chat_template.py).

---

## 1. Overview & Delimiters

Maple is based on the Qwen2 tokenizer vocabulary, using atomic special control tokens:

- `<|im_start|>` (token ID `151644`): Turn opening delimiter.
- `<|im_end|>` (token ID `151645`): Turn closing delimiter.
- `<think>` (token ID `151667`) and `</think>` (token ID `151668`): Delimiters for internal reasoning traces.

Each message turn is serialized as:

```text
<|im_start|>{role}
{content}<|im_end|>
```

When generating an assistant turn (`add_generation_prompt=True`), the template primes both the assistant role and an open thinking tag:

```text
<|im_start|>assistant
<think>
```

---

## 2. Roles & Normalization

- **`system`**: Contains system instructions. If tools are supplied, function schemas are formatted inside `<tools>...</tools>` tags.
    - In `src/maple_run/chat.py` (`normalize_chat_messages`), client messages with `role: "developer"` (sent by tools like Pi/OpenAI) are automatically normalized to `system`.
    - Mid-conversation system messages are hoisted and merged into the root system message so tools and global instructions are never dropped on turn 2+.
- **`user`**: The human/user prompt that initiates a turn.
- **`assistant`**: Assistant outputs, including `<think>` reasoning blocks, textual responses, and `<tool_call>` JSON blocks.
- **`tool`**: Tool execution outputs. In the serialized ChatML prompt, consecutive `tool` messages are wrapped inside a user turn as:
    ```text
    <|im_start|>user
    <tool_response>
    {result}
    </tool_response><|im_end|>
    ```

---

## 3. Reasoning Retention Policies Across Scopes

Reasoning models produce `<think>...</think>` traces before outputting visible content or tool invocations. How these traces are handled across subsequent steps defines the **retention policy**.

There are three primary retention scopes:

| Policy                      | Intra-Turn Tool Calls | Cross User Turns | Configuration                       | Description                                                                                                                          |
| :-------------------------- | :-------------------- | :--------------- | :---------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| **Turn-Scoped** _(Default)_ | **Preserved**         | **Pruned**       | `preserve_thinking=False` (default) | Reasoning is preserved during intermediate tool-call iterations within the active user turn, but pruned once a new user turn begins. |
| **Conversation-Scoped**     | **Preserved**         | **Preserved**    | `preserve_thinking=True`            | All `<think>` traces across the entire dialogue history are kept in the prompt across all turns.                                     |
| **Stateless / Final-Only**  | Pruned                | Pruned           | _(Historical)_                      | All historical `<think>` traces are stripped unconditionally.                                                                        |

---

### Scope A: Conversation-Scoped

- Historical behavior in `chat_template.jinja.orig`.
- **Behavior**: Every past assistant turn retained its `<think>...</think>` block in full.
- **Tradeoffs**:
    - _Pro_: Full history preserved.
    - _Con_: Severe context bloat; past internal monologues distract the model on subsequent turns; prior reasoning often causes the model to anchor to stale assumptions.

---

### Scope B: Stateless / Final-Only

- Behavior in earlier revisions of `chat_template.jinja`.
- **Behavior**: Any historical assistant turn had its reasoning stripped via `content.split('</think>')[-1]`.
- **Tradeoffs**:
    - _Pro_: Minimal context usage; avoids cross-turn reasoning contamination.
    - _Con_: **Breaks agentic workflows**. During a multi-step tool call within the same user turn (e.g. `User request -> Assistant think 1 + Tool call 1 -> Tool response 1 -> Assistant think 2`), the model lost its initial step-1 plan and scratchpad before executing step 2.

---

### Scope C: Turn-Scoped (Maple's Active Policy)

Maple implements **Turn-Scoped** retention in [`checkpoints/maple-2bit/chat_template.jinja`](../checkpoints/maple-2bit/chat_template.jinja).

- **Within the Current User Turn**:
  When an assistant calls a tool, the intermediate reasoning is preserved when presenting the tool response back to the model for its next step. The model retains context on _why_ it called the tool and what it planned to do with the result.
- **Across User Turns**:
  Once the turn completes and the user submits a new prompt (`role: "user"`), all thinking traces from previous turns are pruned from the prompt history. Only the final answers and tool actions survive.

#### Lifecycle Diagram

```
User Turn 1:
  User: "Check the build and fix the error."
    │
    ▼
  Assistant Step 1:
    <think>Planning: I will run `cargo test` first.</think>  ──► [PRESERVED in Step 2]
    <tool_call>bash: cargo test</tool_call>
    │
    ▼
  Tool Response:
    <tool_response>error: missing semicolon</tool_response>
    │
    ▼
  Assistant Step 2:
    (Prompt includes Step 1 <think> planning!)
    <think>The compiler needs a semicolon on line 42. Editing now.</think>
    "Fixed the syntax error."

User Turn 2:
  User: "Now run clippy."
    │
    ▼
  Assistant Turn 2 Prompt:
    Step 1 <think> (cargo test) ─────────► [PRUNED]
    Step 2 <think> (compiler error) ─────► [PRUNED]
    (Only final tool calls and visible answers remain in history)
```

---

## 4. Jinja2 Implementation Details

The turn boundary is determined dynamically by locating the latest user message index:

```jinja
{# Find the index of the latest user message to establish turn boundary #}
{%- set ns = namespace(last_user_idx=-1) %}
{%- for message in messages %}
    {%- if message.role == 'user' %}
        {%- set ns.last_user_idx = loop.index0 %}
    {%- endif %}
{%- endfor %}
```

When iterating over assistant messages:

1. **Intra-turn messages (`loop.index0 > ns.last_user_idx`)**:
   Extracts `reasoning_content` (either from the message dictionary field or parsed from `<think>...</think>` in `content`):
    ```jinja
    {%- if preserve_thinking | default(false) or loop.index0 > ns.last_user_idx %}
        {%- if message.reasoning_content is defined and message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- elif '</think>' in content %}
            {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- set content = content.split('</think>')[-1].lstrip('\n') %}
        {%- endif %}
    ```
2. **Prior-turn messages (`loop.index0 <= ns.last_user_idx`)**:
   Strips `<think>` and discards `reasoning_content`:
    ```jinja
    {%- else %}
        {%- if '</think>' in content %}
            {%- set content = content.split('</think>')[-1].lstrip('\n') %}
        {%- endif %}
    {%- endif %}
    ```
3. **Generation Mask Wrapping**:
   Outputs are wrapped in Hugging Face's `{% generation %}` tags so token masking (`return_assistant_tokens_mask=True`) works correctly for training and evaluation.

---

### The `preserve_thinking` Flag

To switch between turn-scoped and conversation-scoped retention, use the `preserve_thinking` flag (default: `False`):

1. **Jinja Template / Tokenizer**:
    ```python
    prompt = tokenizer.apply_chat_template(
        messages,
        preserve_thinking=True,  # or False (default)
        add_generation_prompt=True,
    )
    ```
2. **HTTP Server API (`/v1/chat/completions`)**:
   Pass `"preserve_thinking": true` in the JSON request body:
    ```json
    {
      "model": "maple-2bit",
      "messages": [...],
      "preserve_thinking": true
    }
    ```
3. **CLI Server Flag**:
   Start the server with conversation-scoped retention as default:
    ```bash
    uv run maple-run serve --model checkpoints/maple-2bit --preserve-thinking
    ```

## 5. Verification Suite

The chat template is verified using [`tests/test_chat_template.py`](../tests/test_chat_template.py):

- `test_preserve_thinking_flag`: Validates `preserve_thinking` flag — `False` (default) prunes historical thoughts; `True` retains thinking traces across all turns in the entire conversation.
- `test_turn_scoped_tool_call_think_preserved`: Verifies intra-turn tool-call reasoning is preserved while cross-turn reasoning is pruned under default `preserve_thinking=False`.
- `test_multi_turn_think_history_stripped`: Verifies standard multi-turn user dialogues do not leak historical thoughts.
- `test_huggingface_assistant_tokens_mask`: Validates official Hugging Face assistant tokens mask computation.
- `test_special_tokens_atomic`: Ensures `<|im_start|>` and `<|im_end|>` encode to single atomic tokens without byte fragmentation.
- `test_generation_prompt_contract`: Confirms `add_generation_prompt=True` ends with `<|im_start|>assistant\n<think>\n`.
- `test_tool_function_unpacking_and_serialization`: Validates both bare and nested `{'type': 'function', 'function': ...}` schemas.
- `test_tool_response_wrapping`: Verifies `<tool_response>` wrapping inside user blocks.

Run the verification suite via:

```bash
uv run pytest tests/test_chat_template.py -v
# Or standalone CLI:
uv run python tests/test_chat_template.py
```
