"""FlashHead clustering + decode: probe clusters, exact logits on those rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from maple_run.cli import main
from maple_run.convert import convert_checkpoint
from maple_run.pack import quantize_rtn

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from tests.test_convert import _make_weights, _tiny_config, _write_json
from tests.test_model import _tiny_config as _tiny_model_config
from tests.test_model import _tiny_weights


def _flash_meta(vocab: int, n_clusters: int, n_probes: int, force=()) -> dict:
    assert vocab % n_clusters == 0
    return {
        "n_clusters": n_clusters,
        "cluster_size": vocab // n_clusters,
        "n_probes": n_probes,
        "group_size": 64,
        "bits": 4,
        "head_group_size": 64,
        "head_bits": 4,
        "scaled_centroids": True,
        "force_tokens": list(force),
    }


def _attach_flash_tensors(
    weights: dict,
    cfg: dict,
    n_clusters: int,
    n_probes: int,
    *,
    force=(),
    rng: np.random.Generator | None = None,
) -> dict:
    vocab = cfg["vocab_size"]
    hidden = cfg["hidden_size"]
    cluster_size = vocab // n_clusters
    token_map = torch.arange(vocab, device="cuda", dtype=torch.int32).reshape(
        n_clusters, cluster_size
    )
    rng = rng or np.random.default_rng(0)
    cw, cs, cb = quantize_rtn(rng.standard_normal((n_clusters, hidden)).astype(np.float32))
    weights["lm_head_flash.token_map"] = token_map
    weights["lm_head_flash.centroids.weight"] = torch.from_numpy(
        np.ascontiguousarray(cw)
    ).to(device="cuda", dtype=torch.uint32)
    weights["lm_head_flash.centroids.scales"] = torch.from_numpy(
        np.ascontiguousarray(cs)
    ).to(device="cuda", dtype=torch.float32)
    weights["lm_head_flash.centroids.biases"] = torch.from_numpy(
        np.ascontiguousarray(cb)
    ).to(device="cuda", dtype=torch.float32)
    meta = _flash_meta(vocab, n_clusters, n_probes, force=force)
    cfg["flash_head"] = meta
    maple = dict(cfg.get("maple_run") or {})
    maple["flash_head"] = meta
    cfg["maple_run"] = maple
    return cfg


@cuda
def test_kmeans_balances_cluster_sizes():
    from maple_run.flash_head import _balanced_spherical_kmeans

    rng = np.random.default_rng(0)
    n, d, n_clusters = 64, 32, 8
    W = rng.standard_normal((n, d)).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    W_t = torch.from_numpy(W).cuda()
    _, labels = _balanced_spherical_kmeans(W_t, n_clusters, n_iter=8, seed=0)
    counts = np.bincount(labels, minlength=n_clusters)
    assert counts.tolist() == [n // n_clusters] * n_clusters


@cuda
def test_flash_head_all_probes_matches_exact_lm_head():
    from maple_run.model import MapleForCausalLM

    cfg = _tiny_model_config()
    rng = np.random.default_rng(1)
    weights = _tiny_weights(cfg, rng)
    n_clusters = 8
    cfg = _attach_flash_tensors(weights, cfg, n_clusters, n_probes=n_clusters, rng=rng)
    exact = MapleForCausalLM.from_weight_dict(
        json.loads(json.dumps(cfg)),
        {k: v.clone() for k, v in weights.items()},
        device="cuda",
        use_flash_head=False,
    )
    flash = MapleForCausalLM.from_weight_dict(
        cfg, weights, device="cuda", use_flash_head=True
    )
    ids = torch.randint(0, cfg["vocab_size"], (1, 1), device="cuda")
    with torch.inference_mode():
        y_exact = exact.forward(ids, cache=exact.make_cache(max_len=8), logits_to_keep=1)
        y_flash = flash.forward(ids, cache=flash.make_cache(max_len=8), logits_to_keep=1)
    assert torch.isfinite(y_flash.float()).all()
    torch.testing.assert_close(y_flash.float(), y_exact.float(), rtol=1e-3, atol=1e-3)


@cuda
def test_flash_head_prefill_stays_on_exact_lm_head():
    from maple_run.model import MapleForCausalLM

    cfg = _tiny_model_config()
    rng = np.random.default_rng(2)
    weights = _tiny_weights(cfg, rng)
    cfg = _attach_flash_tensors(weights, cfg, n_clusters=8, n_probes=2, rng=rng)
    exact = MapleForCausalLM.from_weight_dict(
        json.loads(json.dumps(cfg)),
        {k: v.clone() for k, v in weights.items()},
        device="cuda",
        use_flash_head=False,
    )
    flash = MapleForCausalLM.from_weight_dict(
        cfg, weights, device="cuda", use_flash_head=True
    )
    ids = torch.randint(0, cfg["vocab_size"], (1, 5), device="cuda")
    with torch.inference_mode():
        y_exact = exact.forward(ids, cache=exact.make_cache(max_len=16), logits_to_keep=1)
        y_flash = flash.forward(ids, cache=flash.make_cache(max_len=16), logits_to_keep=1)
    torch.testing.assert_close(y_flash.float(), y_exact.float(), rtol=1e-4, atol=1e-4)


@cuda
def test_flash_head_disabled_ignores_checkpoint_tensors():
    from maple_run.model import MapleForCausalLM

    cfg = _tiny_model_config()
    rng = np.random.default_rng(3)
    weights = _tiny_weights(cfg, rng)
    cfg = _attach_flash_tensors(weights, cfg, n_clusters=8, n_probes=2, rng=rng)
    model = MapleForCausalLM.from_weight_dict(
        cfg, weights, device="cuda", use_flash_head=False
    )
    assert model.lm_head_flash is None
    ids = torch.randint(0, cfg["vocab_size"], (1, 1), device="cuda")
    with torch.inference_mode():
        y = model.forward(ids, cache=model.make_cache(max_len=8), logits_to_keep=1)
    assert y.shape == (1, 1, cfg["vocab_size"])
    assert torch.isfinite(y.float()).all()


def test_generate_flash_head_attaches_shard(tmp_path: Path):
    from maple_run.flash_head import generate_flash_head

    rng = np.random.default_rng(4)
    src = tmp_path / "src"
    dst = tmp_path / "out"
    src.mkdir()
    save_file(_make_weights(rng), str(src / "model.safetensors"))
    _write_json(src / "config.json", _tiny_config(eos_token_id=0))
    convert_checkpoint(str(src), str(dst))

    meta = generate_flash_head(
        str(dst), n_clusters=8, n_iter=4, n_probes=4, seed=0
    )
    assert meta["n_clusters"] == 8
    assert meta["cluster_size"] == 8
    assert meta["n_probes"] == 4
    assert 0 in meta["force_tokens"]
    packed = load_file(str(dst / "model-flashhead.safetensors"))
    assert packed["lm_head_flash.token_map"].shape == (8, 8)
    assert packed["lm_head_flash.centroids.weight"].dtype == np.uint32
    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["flash_head"]["scaled_centroids"] is True
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert index["weight_map"]["lm_head_flash.token_map"] == "model-flashhead.safetensors"
    token_map = packed["lm_head_flash.token_map"]
    assert len(np.unique(token_map)) == 64


def test_cli_flash_head_only(tmp_path: Path):
    rng = np.random.default_rng(5)
    src = tmp_path / "src"
    dst = tmp_path / "out"
    src.mkdir()
    save_file(_make_weights(rng), str(src / "model.safetensors"))
    _write_json(src / "config.json", _tiny_config())
    assert main(["convert", str(src), "-o", str(dst)]) == 0
    assert (
        main(
            [
                "convert",
                str(dst),
                "--flash-head-only",
                "--clusters",
                "8",
                "--probes",
                "4",
                "--kmeans-iters",
                "3",
            ]
        )
        == 0
    )
    assert (dst / "model-flashhead.safetensors").exists()


def test_packed_decode_bytes_flash_head_is_smaller():
    from maple_run.model import packed_decode_bytes

    repo = Path(__file__).resolve().parents[1]
    cfg = json.loads((repo / "docs/sources/maple-preview-config.json").read_text())
    cfg["maple_run"] = {
        "format": "packed-ternary-v1",
        "rtn_4bit": {"group_size": 64, "bits": 4},
        "flash_head": {
            "n_clusters": 4748,
            "cluster_size": 32,
            "n_probes": 512,
            "force_tokens": [151643],
        },
    }
    cfg["flash_head"] = cfg["maple_run"]["flash_head"]
    exact = packed_decode_bytes(cfg, flash_head=False)
    flash = packed_decode_bytes(cfg, flash_head=True)
    assert flash["packed_weight_bytes"] < exact["packed_weight_bytes"]
    assert flash["packed_weight_bytes"] < 0.85 * exact["packed_weight_bytes"]
