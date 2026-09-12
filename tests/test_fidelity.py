"""CPU tests for original-vs-packed fidelity reporting helpers."""

from __future__ import annotations

import importlib.util
import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from maple_run.fidelity import (
    DEFAULT_ORIGINAL,
    FidelityError,
    _fake_flash_attention_imports,
    _legacy_default_rope_init,
    _reset_reference_rope,
    sdpa_attention_forward,
    summarize,
    threshold_outcomes,
    token_metrics,
    run_fidelity,
)


def test_default_original_is_maple_preview():
    assert DEFAULT_ORIGINAL == "deepgrove/maple-preview"


def test_fake_flash_attention_has_an_import_spec():
    with _fake_flash_attention_imports():
        assert importlib.util.find_spec("flash_attn") is not None
        padding = importlib.import_module("flash_attn.bert_padding")
        assert callable(padding.unpad_input)


def test_legacy_default_rope_matches_standard_inverse_frequencies():
    class Config:
        head_dim = 8
        partial_rotary_factor = 0.5
        rope_theta = 10000.0

    got, scale = _legacy_default_rope_init(Config(), device="cpu")
    want = torch.tensor([1.0, 0.01])
    torch.testing.assert_close(got, want)
    assert scale == 1.0


def test_reset_reference_rope_replaces_corrupt_nonpersistent_buffer():
    class Config:
        head_dim = 8
        partial_rotary_factor = 0.5
        rope_theta = 10000.0

    class Rotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.full((2,), float("nan")), persistent=False)
            self.original_inv_freq = self.inv_freq
            self.attention_scaling = 0.0

    model = SimpleNamespace(
        config=Config(), model=SimpleNamespace(rotary_emb=Rotary())
    )
    _reset_reference_rope(model)
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, torch.tensor([1.0, 0.01]))
    assert model.model.rotary_emb.original_inv_freq is model.model.rotary_emb.inv_freq
    assert model.model.rotary_emb.attention_scaling == 1.0


def test_token_metrics_match_hand_computed_distribution():
    original = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
    packed = torch.tensor([[0.0, 0.0], [0.0, 3.0]])
    metrics = token_metrics(original, packed, torch.tensor([0, 0]))
    assert metrics.kl[0] == pytest.approx(0.0)
    p = torch.softmax(original[1], -1)
    q = torch.softmax(packed[1], -1)
    assert metrics.kl[1] == pytest.approx(float((p * (p.log() - q.log())).sum()))
    assert metrics.top1 == [True, False]
    assert metrics.top5 == [True, True]


def test_metrics_are_finite_for_large_logits():
    metrics = token_metrics(
        torch.tensor([[10000.0, -10000.0]]),
        torch.tensor([[-10000.0, 10000.0]]),
        torch.tensor([0]),
    )
    assert metrics.kl[0] == pytest.approx(20000.0)


def test_summary_and_thresholds():
    records = [{
        "kl_values": [0.0, 2.0], "nll_delta_values": [0.0, 1.0],
        "top1_values": [True, False], "top5_values": [True, True],
    }]
    summary = summarize(records)
    assert summary["mean_kl"] == 1.0
    assert summary["p50_kl"] == 1.0
    assert summary["top1_agreement"] == 0.5
    assert not threshold_outcomes(summary, 0.9, 0.5)["passed"]
    assert not threshold_outcomes(summary, 1.0, 0.6)["passed"]
    assert threshold_outcomes(summary, 1.0, 0.5)["passed"]


def test_run_uses_original_tokenizer_as_canonical_ids(tmp_path, monkeypatch):
    class Packed:
        device = torch.device("cpu")

        def forward(self, ids, cache=None, logits_to_keep=0):
            logits = torch.zeros(1, ids.shape[1], 8)
            logits[..., 2] = 1
            return logits

    class Original:
        def __call__(self, input_ids, use_cache=False, logits_to_keep=0):
            logits = torch.zeros(1, input_ids.shape[1], 8)
            logits[..., 2] = 1
            return SimpleNamespace(logits=logits)

    canonical_tokenizer = object()
    seen_tokenizers = []
    monkeypatch.setattr(
        "maple_run.model.MapleForCausalLM.from_packed",
        lambda *_args, **_kwargs: Packed(),
    )
    monkeypatch.setattr(
        "maple_run.fidelity.load_original_reference",
        lambda *_args, **_kwargs: (Original(), canonical_tokenizer),
    )
    monkeypatch.setattr(
        "maple_run.nextlat_synth.iter_user_prompts",
        lambda *_args, **_kwargs: iter(["hello"]),
    )

    def encode(tokenizer, prompt, device):
        seen_tokenizers.append(tokenizer)
        return torch.tensor([[1, 2, 3]], device=device)

    monkeypatch.setattr("maple_run.generate.encode_user_prompt", encode)
    output = tmp_path / "report.json"
    assert run_fidelity("packed", "prompts", output=str(output), log=lambda *_: None)
    assert seen_tokenizers == [canonical_tokenizer]


def _naive_attention(q, k, v, window=None):
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    q_len, k_len = q.shape[-2], k.shape[-2]
    q_pos = torch.arange(q_len) + (k_len - q_len)
    k_pos = torch.arange(k_len)
    allowed = k_pos[None, :] <= q_pos[:, None]
    if window is not None:
        allowed &= q_pos[:, None] - k_pos[None, :] < window
    scores = scores.masked_fill(~allowed.to(scores.device), float("-inf"))
    return torch.softmax(scores, -1) @ v


@pytest.mark.parametrize("window", [None, 3])
def test_sdpa_adapter_matches_naive_causal_gqa(window):
    torch.manual_seed(4)
    q = torch.randn(1, 4, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    module = SimpleNamespace(num_heads=4, num_key_value_heads=2)
    out, weights = sdpa_attention_forward(module, q, k, v, sliding_window=window)
    want = _naive_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), window)
    torch.testing.assert_close(out.transpose(1, 2), want, rtol=1e-5, atol=1e-5)
    assert weights is None


def test_sdpa_adapter_rejects_sequence_major_maple_layout():
    module = SimpleNamespace(num_heads=16, num_key_value_heads=4)
    q = torch.randn(1, 64, 16, 128)
    k = torch.randn(1, 64, 4, 128)
    v = torch.randn(1, 64, 4, 128)
    with pytest.raises(FidelityError, match="wrong layout"):
        sdpa_attention_forward(module, q, k, v)
