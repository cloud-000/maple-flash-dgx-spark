"""Chat-template encoding and think/UTF-8 detokenizing."""

from __future__ import annotations

import torch

from maple_run.generate import (
    CompletionDecoder,
    decode_holding_incomplete,
    encode_messages,
    prompt_in_think,
)


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
