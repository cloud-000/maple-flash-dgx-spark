"""Chat-template encoding and think/UTF-8 detokenizing."""

from __future__ import annotations

import torch

import json
from pathlib import Path

import pytest

from maple_run.generate import (
    CompletionDecoder,
    decode_holding_incomplete,
    encode_messages,
    parse_tool_call_json,
    prompt_in_think,
    split_content_and_tools,
)

PACKED = Path(__file__).resolve().parents[1] / "checkpoints" / "maple-2bit"


class _CapturingTokenizer:
    chat_template = "yes"
    captured_messages = None
    captured_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.captured_messages = messages
        self.captured_kwargs = kwargs
        return torch.tensor([[9, 8, 7]])


class _SeqTokenizer:
    """ids < 100 decode as chr(id); 1/2 are think tags; 10 is a hanging U+FFFD."""

    def get_vocab(self):
        return {"<think>": 1, "</think>": 2}

    def decode(self, ids, skip_special_tokens=False):
        ids = list(ids)
        if ids == [10]:
            return "\ufffd"
        if ids == [10, 11]:
            return "é"
        out = []
        for i in ids:
            if i in (1, 2):
                continue
            out.append("\n" if i == 12 else chr(i))
        return "".join(out)


def test_encode_messages_forwards_tools_and_tool_calls():
    tok = _CapturingTokenizer()
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    messages = [
        {"role": "user", "content": "read foo"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
    ]
    ids = encode_messages(tok, messages, "cpu", tools=tools)
    assert ids.tolist() == [[9, 8, 7]]
    assert tok.captured_kwargs["tools"] == tools
    assert tok.captured_messages[1]["tool_calls"][0]["function"]["name"] == "read_file"


def test_encode_messages_omits_empty_tools():
    tok = _CapturingTokenizer()
    encode_messages(tok, [{"role": "user", "content": "hi"}], "cpu", tools=None)
    assert "tools" not in tok.captured_kwargs


def test_encode_messages_maps_developer_to_system():
    tok = _CapturingTokenizer()
    encode_messages(
        tok,
        [
            {"role": "developer", "content": "You are Bob, a happy builder on mars"},
            {"role": "user", "content": "who are you?"},
        ],
        "cpu",
    )
    assert tok.captured_messages[0]["role"] == "system"
    assert "Bob" in tok.captured_messages[0]["content"]


def test_prompt_in_think_detects_open_tag():
    tok = _SeqTokenizer()
    open_ids = torch.tensor([[5, 1, 10]])
    closed_ids = torch.tensor([[5, 1, 10, 2, 6]])
    assert prompt_in_think(tok, open_ids)
    assert not prompt_in_think(tok, closed_ids)


def test_decoder_splits_think_and_strips_leading_newlines():
    tok = _SeqTokenizer()
    dec = CompletionDecoder(tok, in_think=True, skip_ids=set())
    assert dec.push(ord("A")) == ("A", "")
    assert dec.push(2) == ("", "")  # </think>
    assert dec.push(12) == ("", "")  # leading newline after </think>
    assert dec.push(ord("B")) == ("", "B")
    reasoning, content = dec.finalize()
    assert reasoning == "A"
    assert content == "B"


def test_decoder_holds_incomplete_utf8():
    tok = _SeqTokenizer()
    dec = CompletionDecoder(tok, in_think=False, skip_ids=set())
    assert dec.push(10) == ("", "")
    assert dec.push(11) == ("", "é")
    assert decode_holding_incomplete(tok, [10]) == ""
    assert decode_holding_incomplete(tok, [10, 11]) == "é"


def test_split_content_and_tools():
    text = 'hi\n<tool_call>\n{"name": "read", "arguments": {"path": "x"}}\n</tool_call>\n'
    visible, calls, in_tool = split_content_and_tools(text)
    assert visible == "hi"
    assert not in_tool
    assert calls[0]["function"]["name"] == "read"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "x"}
    open_text = text + "<tool_call>\n{"
    visible, calls, in_tool = split_content_and_tools(open_text)
    assert in_tool
    assert visible == "hi"


def test_parse_tool_call_json_rejects_garbage():
    assert parse_tool_call_json("not json") is None
    assert parse_tool_call_json('{"arguments": {}}') is None


def test_decoder_extracts_tool_call_and_hides_xml():
    tok = _SeqTokenizer()
    dec = CompletionDecoder(tok, in_think=False, skip_ids=set())
    xml = '<tool_call>\n{"name": "read", "arguments": {"path": "x"}}\n</tool_call>'
    for ch in xml:
        dec.push(12 if ch == "\n" else ord(ch))
    assert dec.finalize()[1] == ""
    assert dec.tool_calls[0]["function"]["name"] == "read"
    assert dec.take_new_tool_calls()[0]["type"] == "function"


@pytest.mark.skipif(
    not (PACKED / "tokenizer.json").exists(), reason="packed tokenizer not on disk"
)
def test_maple_template_keeps_system_skills_and_tools():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PACKED, trust_remote_code=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    ids = encode_messages(
        tok,
        [
            {
                "role": "developer",
                "content": "You are Bob, a happy builder who lives on mars.\n\n# Skills\n- Cloudflare Workers",
            },
            {"role": "user", "content": "who are you?"},
        ],
        "cpu",
        tools=tools,
    )
    text = tok.decode(ids[0].tolist(), skip_special_tokens=False)
    assert "You are Bob" in text
    assert "mars" in text
    assert "Cloudflare Workers" in text
    assert "# Tools" in text
    assert "read" in text
    assert "<think>" in text
