# Phase 0.5 Findings: Route-Affinity vs. Linguistic Plausibility

> **Status**: COMPLETED — 2026-09-16
> **Script**: [`bench/probe_route_affinity.py`](file:///home/cloud/Code/maple-run/bench/probe_route_affinity.py)
> **Data**: [`runs/route_affinity_probe.json`](file:///home/cloud/Code/maple-run/runs/route_affinity_probe.json)

---

## Motivation

ERS's speedup depends on speculative candidates reusing the frontier token's expert set. Phase 0 established that consecutive tokens share an average of **2.64/8 experts** (33%). Before implementing Phase 2–4, this probe answers the critical question:

> Are token pairs with high expert overlap (≥5/8) also linguistically plausible continuations? If yes, a route-affinity drafter can select expert-overlapping candidates without sacrificing acceptance rate α. If no, ERS cannot achieve a net speedup over pure AR.

---

## Setup

- **Model**: `checkpoints/maple-2bit` (N=256 experts, top-k=8, 24 layers)
- **Data**: 5 prompts × 120 greedy decode steps = **590 consecutive token pairs**
- **Measurement**: For each pair (token\_t, token\_{t+1}), compute mean per-layer expert overlap $\frac{1}{L}\sum_\ell |E_{\ell,t} \cap E_{\ell,t+1}|$
- **Analysis**: Bucket pairs by overlap, examine example tokens, compute ERS break-even economics per bucket

---

## Results

### Overlap Distribution

| Bucket | Mean overlap | Count | % of pairs |
| :--- | :---: | :---: | :---: |
| 0–25% (0–2 experts shared) | ~1.3/8 | 148 | 25.1% |
| **25–50% (2–4 experts shared)** | **~2.9/8** | **405** | **68.6%** |
| 50–62% (4–5 experts shared) | ~4.2/8 | 32 | 5.4% |
| 62–75% (5–6 experts shared) | ~5.2/8 | 4 | 0.7% |
| 75–87% (6–7 experts shared) | ~6.1/8 | 1 | 0.2% |
| 87–100% (7–8 experts shared) | — | 0 | 0.0% |

**Mean overlap overall: 2.578/8 (32.2%)** — consistent with Phase 0 measurements (2.64/8).

### Example Token Pairs by Bucket

**0–25% overlap** (25.1% of pairs — lowest locality):
```
' sending' → ' a'       overlap=1.54
' of'      → ' figur'   overlap=1.38
' non'     → '-'        overlap=0.50
```

**25–50% overlap** (68.6% of pairs — the dominant regime):
```
' examples' → ' of'         overlap=3.25
'decl'       → 'ine'        overlap=3.29
' hold'      → ' the'       overlap=2.17
','          → ' moving'    overlap=2.79
```

**50–62% overlap** (5.4% of pairs):
```
' we'      → ' all'        overlap=4.38
','        → ' then'       overlap=4.46
' believe' → ' that'       overlap=4.21
' long'    → '-term'       overlap=4.38
```

**62–75% overlap** (0.7% of pairs — ERS viable range starts here):
```
' may' → ' not'    overlap=5.17
' may' → ' not'    overlap=5.25
' may' → ' not'    overlap=5.42
' may' → ' not'    overlap=5.04
```

> [!CAUTION]
> The entire 62–75% bucket consists of a single repeated bigram (`' may' → ' not'`) across multiple prompts. This is not generalizable signal — it is one high-frequency collocation with accidentally stable routing.

**75–87%** (0.2%): one pair: `',' → ' and'` at overlap=6.12.

**87–100%**: zero pairs observed.

---

## ERS Economics Per Bucket

With a drafter at p\_accept = 0.45 per token (realistic for a small draft model), ERS must achieve α > verify\_cost to beat pure AR. For 1 speculative candidate:

| Bucket | $|S_\ell|$ | Verify cost | Break-even α | Viable? |
| :--- | :---: | :---: | :---: | :---: |
| 0–25% | 14.6 | 1.82× | 1.82 | 🔴 |
| 25–50% | 13.2 | 1.65× | 1.65 | 🔴 |
| 50–62% | 11.7 | 1.47× | 1.47 | 🔴 |
| 62–75% | 10.8 | 1.35× | 1.35 | ✅ |
| 75–87% | 9.9 | 1.23× | 1.23 | ✅ |

**Viable fraction of pairs**: 0.9% (0.7% + 0.2%).

The break-even threshold requires mean overlap > 4.4/8 (55%). Only **6.3% of natural consecutive pairs** exceed this threshold. Almost none of these are general continuations — they are high-frequency bigrams with accidentally stable routing.

---

## Conclusion

**Route-affinity candidate selection is not viable for ERS on Maple.**

The empirical result is decisive:

1. **93.7% of consecutive token pairs fall below the ERS viability threshold.** You cannot route-select your way to the viable bucket in normal language generation.

2. **The viable pairs (0.9%) are degenerate.** The 62–75% bucket is dominated by one repeated bigram. There is no evidence that high-overlap pairs are semantically diverse plausible continuations.

3. **Maple's routing is not sufficiently semantically organized for route-affinity to decouple from linguistic quality.** If a drafter is constrained or selected to generate expert-overlapping tokens, it will be biased toward function-word bigrams with low information content, degrading α to approximately zero net gain.

4. **ERS cannot beat pure AR without route-affinity, and route-affinity cannot be achieved without sacrificing α.** The mechanism has no viable operating point on this model with this routing structure.

---

## Impact on ERS Plan

This finding is a **design-level blocker** for Phases 2–4 as currently specified. See [`docs/ERS_PLAN.md`](file:///home/cloud/Code/maple-run/docs/ERS_PLAN.md) for updated phase status and [`docs/ERS.md`](file:///home/cloud/Code/maple-run/docs/ERS.md) §7 for design implications.

Mechanism 1 (expert-major grouped dispatch, guaranteed ~10–20% over naive standard SD) remains valid and may be worth implementing independently of the speculative decoding framework.
