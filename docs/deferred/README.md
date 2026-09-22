# Deferred Research & Experiments

This directory archives experimental tracks and exploratory research that are deferred or failed, keeping the core runtime codebase clean, lean, and maintainable.

## Contents

### 1. [Expert-Reuse Speculation (ERS)](file:///home/cloud/Code/maple-run/docs/deferred/ers/)
- [`ERS.md`](file:///home/cloud/Code/maple-run/docs/deferred/ers/ERS.md): Core thesis, speculative execution theory, state machine, and phase roadmap.
- [`ERS_PLAN.md`](file:///home/cloud/Code/maple-run/docs/deferred/ers/ERS_PLAN.md): Benchmark phase plan and empirical validation criteria.
- [`ERS_PHASE05_FINDINGS.md`](file:///home/cloud/Code/maple-run/docs/deferred/ers/ERS_PHASE05_FINDINGS.md): Empirical findings from Phase 0 and Phase 0.5 probes.
- [`probes/`](file:///home/cloud/Code/maple-run/docs/deferred/ers/probes/): Probe and micro-benchmark scripts:
  - `bench_expert_roofline.py`
  - `probe_expert_cacheability.py`
  - `probe_route_affinity.py`
  - `probe_weight_and_cooccurrence.py`
  - `profile_route_overlap.py`
- [`runs/`](file:///home/cloud/Code/maple-run/docs/deferred/ers/runs/): Probe result JSON outputs.

---

## Pruned Subsystems Note

The following failed / obsolete experimental subsystems have been completely excised from `src/maple_run/`:
- **EGGROLL**: Evolution Strategies (rank-1 adapter perturbation rollout) post-training exploration. Excised from `linear.py`, `generate.py`, and `cli.py`.
- **NextLat**: Latent dynamics self-speculative decoding model ($p_\psi$) and on-policy synthetic trajectory generation. Excised from `generate.py` and `cli.py`. Prompt loading utilities were consolidated into `fidelity.py`.
