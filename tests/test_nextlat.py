"""NextLat latent dynamics, loss, and greedy speculative decode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maple_run.cli import main
from maple_run.nextlat import (
    LatentDynamics,
    accept_drafts,
    default_mlp_hidden,
    nextlat_loss,
    sample_from_probs,
)

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from tests.test_model import _tiny_config, _tiny_weights


def test_default_mlp_hidden_matches_fineweb_ratio():
    assert default_mlp_hidden(2048) == 6528
    assert default_mlp_hidden(128) % 128 == 0


def test_dynamics_residual_and_shapes():
    dyn = LatentDynamics(32, mlp_hidden=128, bias=False)
    h = torch.randn(2, 8, 32)
    e = torch.randn(2, 8, 32)
    y = dyn(h, e)
    assert y.shape == h.shape
    # Zero MLP → identity residual.
    with torch.no_grad():
        for p in dyn.mlp.parameters():
            p.zero_()
    y0 = dyn(h, e)
    torch.testing.assert_close(y0, h)


def test_nextlat_loss_d1_identity_is_shift_distance():
    dyn = LatentDynamics(16, mlp_hidden=64, bias=False)
    with torch.no_grad():
        for p in dyn.mlp.parameters():
            p.zero_()
        if dyn.norm_x.weight is not None:
            dyn.norm_x.weight.fill_(1.0)
    hidden = torch.randn(2, 6, 16)
    embeds = torch.randn(2, 6, 16)
    stats = nextlat_loss(dyn, hidden, embeds, horizon=1, lambda_kl=0.0)
    pred = hidden[:, :-1]  # residual identity
    target = hidden[:, 1:]
    expect = torch.nn.functional.smooth_l1_loss(pred, target)
    torch.testing.assert_close(stats["next_h"], expect)
    assert stats["kl"].item() == 0.0


def test_nextlat_loss_trains_synthetic_shift():
    torch.manual_seed(0)
    hidden_size = 16
    dyn = LatentDynamics(hidden_size, mlp_hidden=64, bias=False)
    opt = torch.optim.AdamW(dyn.parameters(), lr=3e-3)
    # h_{t+1} = h_t + embed(x_{t+1})
    hidden = torch.zeros(4, 8, hidden_size)
    embeds = torch.randn(4, 8, hidden_size)
    for t in range(1, 8):
        hidden[:, t] = hidden[:, t - 1] + embeds[:, t]
    first = None
    last = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        stats = nextlat_loss(
            dyn, hidden.detach(), embeds.detach(), horizon=1, lambda_kl=0.0
        )
        stats["loss"].backward()
        opt.step()
        val = float(stats["next_h"])
        first = val if first is None else first
        last = val
    assert last < first * 0.5, (first, last)


def test_accept_drafts_greedy_all_or_reject():
    gamma = 4
    vocab = 8
    ids = torch.tensor([1, 2, 3, 4])
    q = torch.zeros(gamma, vocab)
    p = torch.zeros(gamma, vocab)
    for i, tok in enumerate(ids.tolist()):
        q[i, tok] = 1.0
        p[i, tok] = 1.0
    bonus = torch.zeros(vocab)
    bonus[5] = 1.0
    n_acc, extra = accept_drafts(ids, q, p, bonus, greedy=True)
    assert n_acc == gamma
    assert extra == 5

    p[2].zero_()
    p[2, 7] = 1.0
    n_acc, extra = accept_drafts(ids, q, p, bonus, greedy=True)
    assert n_acc == 2
    assert extra == 7


def test_sample_from_probs_1d_and_batched():
    row = torch.zeros(8)
    row[3] = 1.0
    tok = sample_from_probs(row)
    assert int(tok.reshape(())) == 3

    batched = row.unsqueeze(0)
    tok_b = sample_from_probs(batched)
    assert tok_b.shape == (1,)
    assert int(tok_b.reshape(())) == 3


def test_accept_drafts_sampled_reject_from_1d_rows():
    # Default generate is sampled. A reject samples a correction from a 1D
    # vocab row; this used to crash in sample_from_probs.
    gamma = 3
    vocab = 8
    ids = torch.tensor([1, 2, 3])
    q = torch.zeros(gamma, vocab)
    p = torch.zeros(gamma, vocab)
    q[0, 1] = 1.0
    p[0, 7] = 1.0  # p(draft)=0 → always reject token 0
    q[1, 2] = 1.0
    p[1, 2] = 1.0
    q[2, 3] = 1.0
    p[2, 3] = 1.0
    bonus = torch.zeros(vocab)
    bonus[4] = 1.0
    n_acc, extra = accept_drafts(ids, q, p, bonus, greedy=False)
    assert n_acc == 0
    assert extra == 7


def test_cli_nextlat_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["nextlat", "--help"])
    assert exc.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "--data" in out
    assert "--horizon" in out
    assert "p_ψ" in out or "latent" in out.lower()


def test_cli_generate_lists_nextlat(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--help"])
    assert exc.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "--nextlat" in out
    assert "--draft-len" in out


def test_cli_nextlat_rejects_bad_steps():
    with pytest.raises(SystemExit):
        main(["nextlat", "--model", "x", "--data", "y", "--steps", "0"])


def test_cli_nextlat_synth_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["nextlat-synth", "--help"])
    assert exc.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "--prompts" in out
    assert "--max-tokens" in out
    assert "--batch-size" in out
    assert "--split" in out


def test_user_prompt_from_mt_bench_list():
    from maple_run.nextlat_synth import user_prompt_from_obj

    assert user_prompt_from_obj({"prompt": ["Write a story", "Rewrite it"]}) == (
        "Write a story"
    )
    assert user_prompt_from_obj({"text": "What is xyz"}) == "What is xyz"
    assert (
        user_prompt_from_obj(
            {
                "messages": [
                    {"role": "user", "content": "Theme version?"},
                    {"role": "assistant", "content": "You are on Responsive 6."},
                ]
            }
        )
        == "Theme version?"
    )


def test_train_jsonl_prefers_input_ids(tmp_path: Path):
    from maple_run.nextlat_train import _read_examples, _tokenize_rows

    path = tmp_path / "onpolicy.jsonl"
    path.write_text(
        json.dumps({"text": "ignore me", "input_ids": [9, 8, 7, 6, 5]}) + "\n"
        + json.dumps({"prompt": "hello world this is long enough"})
        + "\n"
    )
    examples = _read_examples(path)
    assert examples[0]["input_ids"] == [9, 8, 7, 6, 5]

    class _Enc:
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3, 4]

    rows = _tokenize_rows(_Enc(), examples, seq_len=8, device="cpu")
    assert rows[0].tolist() == [9, 8, 7, 6, 5]
    assert rows[1].tolist() == [1, 2, 3, 4]


class _Tok:
    def get_vocab(self):
        return {}

    def decode(self, ids, skip_special_tokens=False):
        return ""

    def encode(self, text, add_special_tokens=False):
        return [3, 4, 5, 6, 7]


@cuda
def test_forward_return_hidden_matches_logits():
    from maple_run.model import MapleForCausalLM

    cfg = _tiny_config()
    rng = __import__("numpy").random.default_rng(0)
    weights = _tiny_weights(cfg, rng)
    model = MapleForCausalLM.from_weight_dict(cfg, weights, device="cuda")
    ids = torch.randint(0, cfg["vocab_size"], (1, 6), device="cuda")
    with torch.inference_mode():
        logits = model.forward(ids, cache=None, logits_to_keep=0)
        logits2, hidden = model.forward(
            ids, cache=None, logits_to_keep=0, return_hidden=True
        )
    assert hidden.shape == (1, 6, cfg["hidden_size"])
    torch.testing.assert_close(logits.float(), logits2.float())
    assert torch.isfinite(hidden.float()).all()


@cuda
def test_greedy_nextlat_matches_greedy_maple():
    from maple_run.generate import run_generate
    from maple_run.model import MapleForCausalLM

    cfg = _tiny_config()
    rng = __import__("numpy").random.default_rng(1)
    weights = _tiny_weights(cfg, rng)
    model = MapleForCausalLM.from_weight_dict(cfg, weights, device="cuda")
    dyn = LatentDynamics(cfg["hidden_size"], mlp_hidden=256, bias=False).to(
        device="cuda", dtype=torch.float32
    )
    ids = torch.randint(1, cfg["vocab_size"], (1, 4), device="cuda")
    tok = _Tok()
    greedy = run_generate(
        model,
        tok,
        ids,
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        use_graph=False,
        log=None,
    )
    spec = run_generate(
        model,
        tok,
        ids,
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        use_graph=False,
        nextlat=dyn,
        draft_len=4,
        log=None,
    )
    assert spec.token_ids == greedy.token_ids
    assert spec.n_new == greedy.n_new


@cuda
def test_sampled_nextlat_generate_does_not_crash():
    from maple_run.generate import run_generate
    from maple_run.model import MapleForCausalLM

    cfg = _tiny_config()
    rng = __import__("numpy").random.default_rng(2)
    weights = _tiny_weights(cfg, rng)
    model = MapleForCausalLM.from_weight_dict(cfg, weights, device="cuda")
    dyn = LatentDynamics(cfg["hidden_size"], mlp_hidden=256, bias=False).to(
        device="cuda", dtype=torch.float32
    )
    ids = torch.randint(1, cfg["vocab_size"], (1, 4), device="cuda")
    result = run_generate(
        model,
        _Tok(),
        ids,
        max_tokens=8,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        use_graph=False,
        nextlat=dyn,
        draft_len=4,
        log=None,
    )
    assert result.n_new >= 1
    assert len(result.token_ids) == result.n_new
    assert result.finish_reason in ("stop", "length")


@cuda
def test_nextlat_save_load_roundtrip(tmp_path: Path):
    dyn = LatentDynamics(32, mlp_hidden=128, bias=False).to("cuda")
    path = tmp_path / "nextlat.safetensors"
    dyn.save(path, extra={"horizon": 1})
    loaded = LatentDynamics.load(path, device="cuda")
    h = torch.randn(1, 4, 32, device="cuda")
    e = torch.randn(1, 4, 32, device="cuda")
    with torch.inference_mode():
        torch.testing.assert_close(loaded(h, e), dyn(h, e))
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["hidden_size"] == 32
    assert meta["horizon"] == 1
