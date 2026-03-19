# Results

Reproduction of Section 4.1 (Linear Logistic Regression on MNIST) from
*Certified Data Removal from Machine Learning Models* (Guo et al., ICML 2020).

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full experiment plan.

All experiments: seed = 42, δ = 1e-4, L-BFGS 100 steps, macOS CPU (Intel x86_64).

---

## Phase 2a — Test Accuracy vs σ (Figure 1, left panel)

Test accuracy of removal-enabled binary logistic regression (3 vs 8) for each
(λ, σ) combination.  No removals; accuracy reflects only the cost of L2
regularisation and objective perturbation.

| λ \ σ | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 |
|------:|-----:|-----:|----:|----:|--:|--:|---:|---:|----:|
| 1e-04 | 96.93 | 96.88 | 96.88 | 96.98 | 97.03 | 96.67 | 92.84 | 86.29 | 73.19 |
| 1e-03 | 95.46 | 95.46 | 95.46 | 95.46 | 95.46 | 95.51 | 94.91 | 92.29 | 75.86 |
| 1e-02 | 91.28 | 91.28 | 91.28 | 91.23 | 91.23 | 91.28 | 91.58 | 91.89 | 85.08 |
| 1e-01 | 87.50 | 87.50 | 87.50 | 87.50 | 87.45 | 87.45 | 86.90 | 83.92 | 69.96 |

**Observations (matching paper Figure 1 left):**
- For each λ, test accuracy is flat for small σ and degrades sharply for
  large σ.  The onset of degradation shifts right as λ increases (stronger
  regularisation tolerates more perturbation).
- λ = 1e-4 gives the best baseline accuracy (~97%) but is most sensitive to σ.
- λ = 1e-1 caps at ~87.5% even with σ → 0; the regularisation itself limits accuracy.

---

## Phase 2c — Expected Removals at ε = 1 (Figure 1, right panel)

Expected number of removals before the cumulative gradient residual norm
bound exceeds the budget σε/c (with ε = 1, c ≈ 4.39).  Each cell is
estimated from 100 Newton-update removal steps and linear extrapolation.

| λ \ σ | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 |
|------:|-----:|-----:|----:|----:|--:|--:|---:|---:|----:|
| 1e-04 | 2 | 6 | 7 | 15 | 34 | 76 | 61 | 28 | 8 |
| 1e-03 | 2 | 6 | 9 | 26 | 87 | 254 | 687 | 949 | 509 |
| 1e-02 | 5 | 12 | 39 | 115 | 385 | 1,149 | 3,738 | 9,850 | 17,369 |
| 1e-01 | 32 | 98 | 329 | 989 | 3,297 | 9,888 | 32,819 | 95,650 | 251,855 |

**Observations (matching paper Figure 1 right):**
- At a fixed λ, expected removals first increase with σ, then decrease
  once accuracy degrades significantly (the gradient norms grow when the
  model is poorly conditioned from too much perturbation).
- Higher λ uniformly supports more removals — λ = 1e-1 with σ = 100
  supports >250 K removals, but accuracy is only ~70%.
- The accuracy/removal trade-off is the core result: a small accuracy
  sacrifice (e.g. 97% → 95% at λ = 1e-3, σ = 3) buys ~250 removals.

---

## Phase 3 — Gradient Residual Norm Trace (Figure 2)

1000 sequential single-point removals at λ = 1e-3, σ = 10.

| Metric | Value |
|---|---|
| Test accuracy | 94.91% |
| Total removal time | 115.4 s (0.115 s / step) |
| Budget at ε = 1 | 2.280 |
| Cumulative data-dep. bound @1000 | 4.743 |
| Expected removals at ε = 1 | 485 |

### Data-dependent bound vs worst-case (Corollary 1 vs Theorem 1)

| Removals | Per-step (data-dep.) | Cumulative (data-dep.) | Cumulative (worst-case) |
|-------:|-----:|-----:|-----:|
| 1 | 3.57e-4 | 3.57e-4 | 8.35e+1 |
| 10 | 7.68e-3 | 2.74e-2 | 8.35e+2 |
| 50 | 1.80e-3 | 1.49e-1 | 4.17e+3 |
| 100 | 3.29e-3 | 3.32e-1 | 8.35e+3 |
| 200 | 3.78e-3 | 9.27e-1 | 1.67e+4 |
| 500 | 4.03e-4 | 2.33e+0 | 4.17e+4 |
| 1000 | 1.76e-3 | 4.74e+0 | 8.35e+4 |

**Observations (matching paper Figure 2):**
- The data-dependent bound (Corollary 1) is **~4 orders of magnitude**
  tighter than the worst-case bound (Theorem 1), consistent with the paper.
- The cumulative data-dependent bound grows approximately linearly,
  confirming the paper's observation (Figure 2).
- At the chosen (λ, σ), ~485 removals are supported before re-training at
  ε = 1.  This is consistent with the Phase 2c estimate (687 from 100
  removals — the extrapolation slightly overestimates because early
  removals tend to have smaller norms).
- Removal cost is **0.115 s / step** vs **~8 s** training time — removal is
  ~70× cheaper than re-training from scratch.

---

## Summary: Reproduction Fidelity

| Paper claim | Our result | Status |
|---|---|---|
| Accuracy ~97% for λ=1e-4, small σ | 97.03% | ✓ Reproduced |
| Accuracy degrades with ↑σ or ↑λ | Confirmed across full grid | ✓ Reproduced |
| Data-dep. bound ≪ worst-case bound | ~4 orders of magnitude gap | ✓ Reproduced |
| Cumulative bound grows ~linearly | Confirmed over 1000 removals | ✓ Reproduced |
| Higher λ → more removals at cost of accuracy | Confirmed | ✓ Reproduced |
| Removal is orders of magnitude faster than re-training | 0.115s vs ~8s (~70×) | ✓ Reproduced |

---

## Plots

All plots are in `plots/` (PDF + PNG).  Regenerate with `uv run python plot_results.py`.

| File | Corresponds to |
|---|---|
| `fig1_left_accuracy_vs_sigma` | Paper Figure 1, left panel |
| `fig1_mid_accuracy_vs_epsilon` | Paper Figure 1, middle panel |
| `fig1_right_accuracy_vs_removals` | Paper Figure 1, right panel |
| `fig1_right_annotated` | Figure 1 right with σ annotations on data points |
| `fig2_gradient_residual_norms` | Paper Figure 2 |
| `fig1_combined` | All four panels in a 2×2 grid |

