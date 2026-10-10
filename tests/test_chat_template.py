"""Chat template verification suite and Hugging Face built-in validator tests.

Can be run via pytest:
    uv run pytest tests/test_chat_template.py -v
    uv run pytest tests/test_chat_template.py --template-file checkpoints/maple-2bit/chat_template.jinja.broken -v
    uv run pytest tests/test_chat_template.py --template-text "{% for m in messages %}{{ m.content }}{% endfor %}" -v

Or run directly as a CLI script:
    uv run python tests/test_chat_template.py
    uv run python tests/test_chat_template.py checkpoints/maple-2bit/chat_template.jinja.orig
    uv run python tests/test_chat_template.py --text "{% for m in messages %}{{ m.content }}{% endfor %}"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pytest
from transformers import AutoTokenizer


CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "maple-2bit"


@pytest.fixture(scope="module")
def tokenizer(request):
    if not CHECKPOINT_DIR.exists():
        pytest.skip(f"Checkpoint directory {CHECKPOINT_DIR} not found.")
    tok = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))

    # Check for CLI overrides from pytest
    template_file = request.config.getoption("--template-file", default=None)
    template_text = request.config.getoption("--template-text", default=None)

    if template_file:
        path = Path(template_file)
        if not path.is_file():
            pytest.fail(f"Specified template file does not exist: {template_file}")
        tok.chat_template = path.read_text(encoding="utf-8")
    elif template_text:
        tok.chat_template = template_text

    return tok


def test_huggingface_assistant_tokens_mask(tokenizer):
    """Test Hugging Face's official assistant token mask validator (return_assistant_tokens_mask=True)."""
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Can you list the repository files?"},
        {
            "role": "assistant",
            "content": "I will inspect the directory.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls -la"}'},
                }
            ],
        },
    ]

    result = tokenizer.apply_chat_template(
        messages,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )

    assert "assistant_masks" in result, "Result must contain assistant_masks"
    input_ids = result["input_ids"]
    assistant_masks = result["assistant_masks"]
    assert len(input_ids) == len(assistant_masks), "input_ids and assistant_masks must match in length"

    # Assistant mask must contain 1s for the assistant generation
    assert sum(assistant_masks) > 0, "HF validator failed: assistant_masks sum is 0!"

    # Verify that the decoded assistant tokens match the assistant's generation
    assistant_token_ids = [
        tok_id for tok_id, mask in zip(input_ids, assistant_masks) if mask == 1
    ]
    decoded_assistant = tokenizer.decode(assistant_token_ids)
    assert "I will inspect the directory." in decoded_assistant
    assert "<tool_call>" in decoded_assistant
    assert '{"command": "ls -la"}' in decoded_assistant

    # Verify that system and user prompt tokens are NOT masked as assistant (mask == 0)
    prompt_token_ids = [
        tok_id for tok_id, mask in zip(input_ids, assistant_masks) if mask == 0
    ]
    decoded_prompt = tokenizer.decode(prompt_token_ids)
    assert "You are a helpful coding assistant." in decoded_prompt
    assert "Can you list the repository files?" in decoded_prompt


def test_special_tokens_atomic(tokenizer):
    """Verify core ChatML delimiters encode to atomic single token IDs without fragmentation."""
    delimiters = ["<|im_start|>", "<|im_end|>"]
    for delimiter in delimiters:
        token_ids = tokenizer.encode(delimiter, add_special_tokens=False)
        assert len(token_ids) == 1, (
            f"Delimiter {delimiter} fragmented into {len(token_ids)} tokens: {token_ids}"
        )


def test_generation_prompt_contract(tokenizer):
    """Verify add_generation_prompt contract: True primes assistant+think; False closes turn."""
    messages = [{"role": "user", "content": "Hello!"}]

    with_gen = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert with_gen.endswith("<|im_start|>assistant\n<think>\n"), (
        f"add_generation_prompt=True must end with '<|im_start|>assistant\\n<think>\\n', got {repr(with_gen[-40:])}"
    )

    without_gen = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    assert without_gen.endswith("<|im_end|>\n"), (
        f"add_generation_prompt=False must end with '<|im_end|>\\n', got {repr(without_gen[-20:])}"
    )


def test_no_escaped_newlines(tokenizer):
    """Verify template does not leak literal '\\n' backslash strings instead of real newlines."""
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "User prompt."},
        {"role": "assistant", "content": "Assistant response."},
        {"role": "tool", "content": "Tool output."},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert r"\n" not in rendered, (
        "Rendered prompt contains literal '\\n' backslash string! Check template for double-escaped newlines."
    )


def test_multi_turn_think_history_stripped(tokenizer):
    """Verify prior assistant reasoning (<think>...</think>) is stripped from conversation history."""
    messages = [
        {"role": "user", "content": "Turn 1 request"},
        {
            "role": "assistant",
            "content": "<think>\nInternal reasoning that should never leak into Turn 2 prompt.\n</think>\nAction result.",
        },
        {"role": "user", "content": "Turn 2 follow-up"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    assert "Internal reasoning that should never leak into Turn 2 prompt." not in rendered, (
        "Historical reasoning trace leaked into multi-turn prompt history!"
    )
    assert "Action result." in rendered, "Assistant output action should be preserved."
    assert "Turn 2 follow-up" in rendered, "Latest user turn should be preserved."


def test_tool_function_unpacking_and_serialization(tokenizer):
    """Verify tool schemas with both wrapped {'type': 'function', 'function': {...}} and bare dicts format correctly."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read file contents.",
                "parameters": {
                    "type": "object",
                    "required": ["path"],
                    "properties": {"path": {"type": "string"}},
                },
            },
        },
        {
            "name": "bash",
            "description": "Execute bash command.",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {"command": {"type": "string"}},
            },
        },
    ]

    messages = [{"role": "user", "content": "Read the file"}]
    rendered = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )

    assert "<tools>" in rendered and "</tools>" in rendered
    assert '"name": "read"' in rendered
    assert '"name": "bash"' in rendered
    assert '"type": "function", "function":' not in rendered


def test_tool_response_wrapping(tokenizer):
    """Verify tool execution outputs are wrapped in <tool_response> tags under a user turn."""
    messages = [
        {"role": "user", "content": "List files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ],
        },
        {"role": "tool", "content": "README.md\npyproject.toml"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    assert "<tool_response>\nREADME.md\npyproject.toml\n</tool_response>" in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")


# ---------------------------------------------------------------------------
# Standalone CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run chat template verification suite on a template file or raw string."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Optional path to template file or raw template string",
    )
    parser.add_argument(
        "-f", "--file",
        dest="file_path",
        default=None,
        help="Path to chat template file to test",
    )
    parser.add_argument(
        "-t", "--text",
        dest="template_text",
        default=None,
        help="Raw chat template string to test",
    )
    parser.add_argument(
        "--model",
        default=str(CHECKPOINT_DIR),
        help=f"Tokenizer checkpoint directory (default: {CHECKPOINT_DIR})",
    )

    args = parser.parse_args()

    # Determine input template
    template_str = None
    target_desc = "active default template"

    if args.file_path:
        p = Path(args.file_path)
        if not p.is_file():
            print(f"Error: file not found: {p}", file=sys.stderr)
            return 2
        template_str = p.read_text(encoding="utf-8")
        target_desc = f"file: {p}"
    elif args.template_text:
        template_str = args.template_text
        target_desc = "inline text"
    elif args.target:
        p = Path(args.target)
        if p.is_file():
            template_str = p.read_text(encoding="utf-8")
            target_desc = f"file: {p}"
        else:
            template_str = args.target
            target_desc = "inline text"

    tok = AutoTokenizer.from_pretrained(args.model)
    if template_str is not None:
        tok.chat_template = template_str

    print(f"\nRunning Chat Template Verification Suite on [{target_desc}]...")
    print("-" * 75)

    all_tests = [
        ("HF Assistant Mask ({% generation %})", test_huggingface_assistant_tokens_mask),
        ("Atomic Special Tokens", test_special_tokens_atomic),
        ("Generation Prompt Contract", test_generation_prompt_contract),
        ("Clean Newlines (no literal \\n)", test_no_escaped_newlines),
        ("Prune Prior <think> from History", test_multi_turn_think_history_stripped),
        ("Tool Function Schema Unpacking", test_tool_function_unpacking_and_serialization),
        ("Tool Response Wrapping", test_tool_response_wrapping),
    ]

    passed_count = 0
    for name, test_fn in all_tests:
        try:
            test_fn(tok)
            print(f"  [PASS] {name}")
            passed_count += 1
        except Exception as e:
            err_msg = str(e).split("\n")[0]
            print(f"  [FAIL] {name}: {err_msg}")

    print("-" * 75)
    total = len(all_tests)
    print(f"Result: {passed_count}/{total} passed.\n")

    return 0 if passed_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
