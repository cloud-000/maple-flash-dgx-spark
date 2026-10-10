"""Chat message formatting, prompt encoding, tool call parsing, and streaming completion."""

from __future__ import annotations

import json
import re
import uuid

import torch

_REPLACEMENT = "\ufffd"
_THINK_START = "<think>"
_THINK_END = "</think>"
_TOOL_START = "<tool_call>"
_TOOL_END = "</tool_call>"
_TOOL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


def message_text(content) -> str:
    """Flatten OpenAI message content (string or text parts) to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type", "text") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                raise ValueError("Only text message content is supported.")
        return "".join(parts)
    raise ValueError("message content must be a string or a list of text parts.")


def _as_token_id_set(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value if v is not None}
    return {int(value)}


def stop_token_ids(model, tokenizer=None) -> set[int]:
    """Token ids that end a response (``<|im_end|>`` and pad/EOT)."""
    ids = _as_token_id_set(getattr(model, "eos_token_id", None))
    if tokenizer is not None:
        ids |= _as_token_id_set(getattr(tokenizer, "eos_token_id", None))
        ids |= _as_token_id_set(getattr(tokenizer, "pad_token_id", None))
    return ids


def think_token_ids(tokenizer) -> tuple[int | None, int | None]:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    return vocab.get(_THINK_START), vocab.get(_THINK_END)


def skip_decode_ids(tokenizer, extra=None) -> set[int]:
    """Special / control tokens that must not appear in streamed text."""
    ids = _as_token_id_set(extra)
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        ids |= _as_token_id_set(getattr(tokenizer, attr, None))
    extra_special = getattr(tokenizer, "additional_special_tokens", None) or []
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        for token in extra_special:
            tid = convert(token)
            if isinstance(tid, int) and tid >= 0:
                ids.add(tid)
    start, end = think_token_ids(tokenizer)
    ids |= {tid for tid in (start, end) if tid is not None}
    return ids


def prompt_in_think(tokenizer, input_ids: torch.Tensor) -> bool:
    """True when the chat template left the prompt inside an open ``<think>``."""
    start, end = think_token_ids(tokenizer)
    if start is None:
        return False
    last_s = last_e = -1
    for i, tid in enumerate(input_ids.reshape(-1).tolist()):
        if tid == start:
            last_s = i
        elif end is not None and tid == end:
            last_e = i
    return last_s > last_e


def decode_holding_incomplete(tokenizer, token_ids: list[int]) -> str:
    """Decode, but hold a trailing U+FFFD so incomplete UTF-8 is not emitted."""
    if not token_ids:
        return ""
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    if text.endswith(_REPLACEMENT):
        text = text[:-1]
    return text


def parse_tool_call_json(body: str) -> dict | None:
    """Parse Maple ``<tool_call>`` JSON into an OpenAI tool_calls item."""
    try:
        obj = json.loads(body.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not obj.get("name"):
        return None
    args = obj.get("arguments", {})
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": str(obj["name"]), "arguments": args},
    }


def split_content_and_tools(text: str) -> tuple[str, list[dict], bool]:
    """Strip ``<tool_call>`` blocks from visible text. ``in_tool`` if one is open."""
    calls: list[dict] = []
    for match in _TOOL_RE.finditer(text):
        parsed = parse_tool_call_json(match.group(1))
        if parsed is not None:
            calls.append(parsed)
    last_start = text.rfind(_TOOL_START)
    last_end = text.rfind(_TOOL_END)
    in_tool = last_start >= 0 and last_start > last_end
    cut = last_start if in_tool else None
    visible = text if cut is None else text[:cut]
    visible = _TOOL_RE.sub("", visible)
    return visible.strip("\n"), calls, in_tool


class CompletionDecoder:
    """Split Maple ``<think>`` traces and hold incomplete UTF-8 across tokens.

    Maple's chat template always opens ``<think>`` on the generation prompt, so
    the completion starts mid-thought. mlx-lm puts that span in ``reasoning``
    and the answer in ``content``; dumping both into ``content`` (and emitting
    U+FFFD on partial Qwen bytes) is what agent harnesses see as garbage.
    """

    def __init__(
        self,
        tokenizer,
        *,
        in_think: bool,
        skip_ids: set[int] | None = None,
    ):
        self.tokenizer = tokenizer
        self.think_start, self.think_end = think_token_ids(tokenizer)
        self.skip_ids = skip_decode_ids(tokenizer) if skip_ids is None else set(skip_ids)
        self.state = "think" if in_think else "content"
        self._think_ids: list[int] = []
        self._content_ids: list[int] = []
        self._think_text = ""
        self._content_text = ""
        self.tool_calls: list[dict] = []
        self._emitted_tools = 0
        self.should_stop = False

    def take_new_tool_calls(self) -> list[dict]:
        new = self.tool_calls[self._emitted_tools :]
        self._emitted_tools = len(self.tool_calls)
        return new

    def push(self, token_id: int) -> tuple[str, str]:
        """Feed one token. Returns ``(reasoning_delta, content_delta)``."""
        tid = int(token_id)
        if self.think_start is not None and tid == self.think_start:
            if self.tool_calls:
                self.should_stop = True
                return "", ""
            self.state = "think"
            return "", ""
        if self.think_end is not None and tid == self.think_end:
            self.state = "content"
            return "", ""
        if tid in self.skip_ids:
            return "", ""
        if self.state == "think":
            self._think_ids.append(tid)
            text = decode_holding_incomplete(self.tokenizer, self._think_ids)
            delta = text[len(self._think_text) :]
            self._think_text = text
            return delta, ""
        self._content_ids.append(tid)
        raw = decode_holding_incomplete(self.tokenizer, self._content_ids)
        visible, calls, in_tool = split_content_and_tools(raw)
        if self.tool_calls and not in_tool and visible[len(self._content_text) :].strip():
            self.should_stop = True
        self.tool_calls = calls
        delta = visible[len(self._content_text) :]
        self._content_text = visible
        return "", delta

    def finalize(self) -> tuple[str, str]:
        return self._think_text, self._content_text


def _message_role(message: dict) -> str:
    role = str(message.get("role") or "")
    return "system" if role == "developer" else role


def normalize_chat_messages(messages: list[dict]) -> list[dict]:
    """Hoist ``system`` / ``developer`` to the front and merge into one block.

    Maple's jinja only injects custom instructions into the tools/system header
    when ``messages[0]`` is system. Agent clients often drop system on turn 2+ or
    send user before system; merging fixes both without changing other turns.
    """
    system_parts: list[str] = []
    rest: list[dict] = []
    for message in messages:
        if _message_role(message) == "system":
            text = message_text(message.get("content"))
            if text.strip():
                system_parts.append(text)
        else:
            rest.append(message)
    if not system_parts:
        return list(messages)
    merged = {"role": "system", "content": "\n\n".join(system_parts)}
    return [merged, *rest]


def _tool_function_name(tool: dict) -> str | None:
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        return str(name) if name else None
    name = tool.get("name")
    return str(name) if name else None


def _stub_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def infer_tools_from_messages(messages: list[dict]) -> list[dict]:
    """Minimal OpenAI tool stubs from assistant ``tool_calls`` already in history."""
    names: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if str(message.get("role")) != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            if not isinstance(fn, dict):
                fn = call if isinstance(call, dict) else {}
            name = fn.get("name")
            if not name:
                continue
            name = str(name)
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
    return [_stub_tool(name) for name in names]


def merge_tools(
    explicit: list | None, inferred: list | None
) -> list[dict] | None:
    """Union tool lists by name; explicit schemas win over history stubs."""
    if not explicit and not inferred:
        return None
    by_name: dict[str, dict] = {}
    for tool in inferred or []:
        if not isinstance(tool, dict):
            continue
        name = _tool_function_name(tool)
        if name:
            by_name[name] = tool
    for tool in explicit or []:
        if not isinstance(tool, dict):
            continue
        name = _tool_function_name(tool)
        if name:
            by_name[name] = tool
    return list(by_name.values()) if by_name else None


def resolve_chat_tools(
    messages: list[dict], tools: list | None
) -> tuple[list[dict] | None, str | None]:
    """Parse request tools, infer missing names from history, and merge."""
    inferred = infer_tools_from_messages(messages)
    merged = merge_tools(tools, inferred)
    if merged is None:
        return None, None
    if not tools and inferred:
        return merged, (
            f"inferred {len(inferred)} tool(s) from history; "
            "re-send tools for full schemas"
        )
    if tools and inferred and len(merged) > len(tools):
        added = len(merged) - len(tools)
        return merged, f"merged request tools with {added} name(s) from history"
    return merged, None


def _chat_message(message: dict) -> dict:
    """Keep tool_calls / reasoning / names; flatten multimodal content to text.

    Pi sends the system prompt as ``developer`` when the model is marked
    reasoning + ``supportsDeveloperRole``. Maple's jinja only reads ``system``.
    """
    out = dict(message)
    out["role"] = _message_role(message)
    out["content"] = message_text(message.get("content"))
    return out


def _to_input_ids(encoded, device) -> torch.Tensor:
    input_ids = encoded if torch.is_tensor(encoded) else encoded["input_ids"]
    return input_ids.to(device).long()


def encode_messages(
    tokenizer,
    messages: list[dict],
    device,
    *,
    tools: list | None = None,
    preserve_thinking: bool = False,
) -> torch.Tensor:
    """Apply the chat template, or concatenate roles if the tokenizer has none.

    Passes ``tools`` and keeps ``tool_calls`` so agent turns match mlx-lm."""
    cleaned = [_chat_message(m) for m in normalize_chat_messages(messages)]
    if getattr(tokenizer, "chat_template", None):
        kwargs: dict = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        if preserve_thinking:
            kwargs["preserve_thinking"] = True
        if tools:
            kwargs["tools"] = tools
        encoded = tokenizer.apply_chat_template(cleaned, **kwargs)
        return _to_input_ids(encoded, device)
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in cleaned)
    return _to_input_ids(
        tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt"),
        device,
    )


def describe_chat_prompt(
    tokenizer,
    input_ids: torch.Tensor,
    tools: list | None,
    *,
    tools_note: str | None = None,
) -> str:
    """One-line serve log: system/tools presence in the tokenized prompt."""
    ids = input_ids.reshape(-1)
    n = int(ids.numel())
    text = tokenizer.decode(ids.reshape(-1).tolist(), skip_special_tokens=False)
    system = "<|im_start|>system" in text
    names = [_tool_function_name(t) for t in tools or []]
    names = [name for name in names if name]
    head = tokenizer.decode(ids[: min(n, 96)].tolist(), skip_special_tokens=False)
    head = " ".join(head.split())
    if len(head) > 120:
        head = head[:120] + "…"
    parts = [
        f"Chat prompt {n} tok",
        f"system={system}",
        f"tools={len(names)}",
    ]
    if names:
        preview = ", ".join(names[:6])
        if len(names) > 6:
            preview += f", +{len(names) - 6} more"
        parts.append(f"tool_names=[{preview}]")
    if tools_note:
        parts.append(f"({tools_note})")
    parts.append(f"head={head!r}")
    return " ".join(parts)


def encode_prompt(tokenizer, prompt: str, device) -> torch.Tensor:
    """Encode a raw completion prompt (no chat template)."""
    return _to_input_ids(
        tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt"),
        device,
    )


def encode_user_prompt(tokenizer, prompt: str, device) -> torch.Tensor:
    """Encode a CLI-style user string through the chat template."""
    return encode_messages(
        tokenizer, [{"role": "user", "content": prompt}], device
    )
