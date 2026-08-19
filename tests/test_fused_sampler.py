"""The two-launch Triton sampler must agree with the torch reference chain."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

VOCAB = 151936


def _reference_probs(logits, temperature, top_p, top_k):
    """Same math as ``sample_next``, but returns the distribution it samples from."""
    from maple_run.generate import _nucleus

    vals, idx = torch.topk(logits, top_k, dim=-1)
    probs = torch.softmax(vals.float().div(temperature), dim=-1)
    if 0.0 < top_p < 1.0:
        probs = _nucleus(probs, top_p)
        probs = probs / probs.sum().clamp_min(1e-12)
    return probs, idx


@cuda
@pytest.mark.parametrize("top_k", [1 + 1, 20, 64])
@pytest.mark.parametrize("top_p", [0.95, 1.0])
def test_fused_sample_inverts_the_same_cdf(top_k, top_p):
    """For a given uniform, the fused kernel picks the reference CDF's token."""
    from maple_run.kernels.sampler import fused_sample

    torch.manual_seed(0)
    temperature = 0.8
    for trial in range(8):
        # float32: bf16 logits collide often enough over 152k values that torch
        # and the kernel break ties differently. Either choice is a correct
        # top-k, so pinning one here would test tie order, not the sampler.
        logits = torch.randn(VOCAB, device="cuda") * 3.0
        probs, idx = _reference_probs(logits, temperature, top_p, top_k)
        cdf = probs.cumsum(dim=-1)
        for u_val in (0.0, 0.25, 0.5, 0.75, 0.999):
            u = torch.tensor(u_val, device="cuda")
            got = fused_sample(
                logits, u, temperature=temperature, top_p=top_p, top_k=top_k
            )
            want = idx[int((cdf <= u_val * cdf[-1]).sum().clamp(max=top_k - 1))]
            assert int(got) == int(want), (trial, u_val, int(got), int(want))


@cuda
def test_fused_sample_respects_the_nucleus():
    """Tokens outside top_p must never be drawn, even at u -> 1."""
    from maple_run.kernels.sampler import fused_sample

    torch.manual_seed(1)
    logits = torch.full((VOCAB,), -20.0, device="cuda")
    logits[7] = 10.0  # ~all the mass
    logits[11] = 2.0
    logits[13] = 1.0
    drawn = set()
    for i in range(64):
        u = torch.tensor(i / 64.0, device="cuda")
        drawn.add(
            int(fused_sample(logits, u, temperature=1.0, top_p=0.9, top_k=20))
        )
    assert drawn == {7}, drawn


@cuda
def test_fused_sample_matches_reference_distribution():
    """Sampling frequencies track the reference probabilities."""
    from maple_run.kernels.sampler import fused_sample

    torch.manual_seed(2)
    logits = torch.randn(VOCAB, device="cuda") * 2.0
    probs, idx = _reference_probs(logits, 1.0, 0.95, 20)

    counts = torch.zeros(20, device="cuda")
    n = 4000
    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    for _ in range(n):
        u = torch.rand((), device="cuda", generator=gen)
        tok = int(fused_sample(logits, u, temperature=1.0, top_p=0.95, top_k=20))
        counts[int((idx == tok).nonzero()[0, 0])] += 1

    freq = (counts / n).cpu()
    assert torch.allclose(freq, probs.cpu(), atol=0.02), (freq, probs.cpu())


@cuda
def test_fused_sampler_rejects_unsupported_settings():
    from maple_run.generate import can_fuse_sampler
    from maple_run.kernels.sampler import MAX_TOP_K, fused_sample

    assert can_fuse_sampler(1.0, 0.95, 20)
    assert not can_fuse_sampler(0.0, 0.95, 20)  # greedy
    assert not can_fuse_sampler(1.0, 0.95, 1)  # greedy
    assert not can_fuse_sampler(1.0, 0.95, 0)  # full vocab, no top-k
    assert not can_fuse_sampler(1.0, 0.95, MAX_TOP_K + 1)

    logits = torch.randn(VOCAB, device="cuda")
    u = torch.tensor(0.5, device="cuda")
    with pytest.raises(ValueError):
        fused_sample(logits, u, temperature=1.0, top_p=0.95, top_k=0)
    with pytest.raises(ValueError):
        fused_sample(logits, u, temperature=0.0, top_p=0.95, top_k=20)


@cuda
@pytest.mark.parametrize("top_k", [1, 20, 64])
@pytest.mark.parametrize("top_p", [0.3, 0.95, 1.0])
def test_batched_sampler_matches_per_row(top_k, top_p):
    """Row ``b`` must draw exactly what it would have drawn alone.

    The batch is only a grid dimension in these kernels, so the bar is bitwise
    equality, not distributional agreement.
    """
    from maple_run.kernels.sampler import fused_sample, fused_sample_batched

    torch.manual_seed(4)
    bsz = 9
    logits = torch.randn(bsz, VOCAB, device="cuda") * 4
    u = torch.rand(bsz, device="cuda")
    got = fused_sample_batched(
        logits, u, temperature=0.8, top_p=top_p, top_k=top_k
    )
    want = torch.stack(
        [
            fused_sample(
                logits[b].contiguous(), u[b], temperature=0.8, top_p=top_p, top_k=top_k
            )
            for b in range(bsz)
        ]
    )
    assert torch.equal(got, want)


@cuda
def test_batched_sampler_reads_rows_of_a_strided_view():
    """Callers pass ``logits[:, -1, :]``, which is not contiguous."""
    from maple_run.kernels.sampler import fused_sample, fused_sample_batched

    torch.manual_seed(6)
    bsz = 5
    logits = torch.randn(bsz, 3, VOCAB, device="cuda") * 4
    u = torch.rand(bsz, device="cuda")
    got = fused_sample_batched(
        logits[:, -1, :], u, temperature=1.0, top_p=0.95, top_k=20
    )
    want = torch.stack(
        [
            fused_sample(
                logits[b, -1, :].contiguous(),
                u[b],
                temperature=1.0,
                top_p=0.95,
                top_k=20,
            )
            for b in range(bsz)
        ]
    )
    assert torch.equal(got, want)


@cuda
def test_batched_sampler_workspace_is_reused_not_resized():
    """A narrower replay slices the widest workspace; it must not reallocate."""
    from maple_run.kernels.sampler import batched_sampler_workspace, fused_sample_batched

    torch.manual_seed(8)
    ws = batched_sampler_workspace(VOCAB, 20, 8, "cuda")
    ptrs = [t.data_ptr() for t in ws]
    logits = torch.randn(8, VOCAB, device="cuda") * 4
    u = torch.rand(8, device="cuda")
    wide = fused_sample_batched(
        logits, u, temperature=1.0, top_p=0.95, top_k=20, workspace=ws
    ).clone()
    narrow = fused_sample_batched(
        logits[:3], u[:3], temperature=1.0, top_p=0.95, top_k=20, workspace=ws
    )
    assert [t.data_ptr() for t in ws] == ptrs
    assert torch.equal(narrow, wide[:3])
