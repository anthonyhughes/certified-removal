## Non-Monotonic Spikes in Accuracy vs Expected Removals

The "Accuracy vs Removals (ε=1)" plot exhibits dramatic downward spikes — particularly visible for the λ=10⁻⁴ curve. These are **not** numerical errors; they arise from a non-monotonic relationship between σ and expected removals when using empirical gradient norms, combined with the plot sorting points by expected removals on the x-axis.

### Root Cause

Consider the λ=10⁻⁴ data:

| σ | Accuracy | E[removals] | Cumul. grad norm |
|---|----------|-------------|------------------|
| 0.3 | 96.98% | 15 | 0.68 |
| 1 | 97.03% | 34 | 0.73 |
| 3 | 96.67% | 76 | 1.06 |
| 10 | 92.84% | 61 | 4.87 |
| 30 | 86.29% | 28 | 45.47 |
| 100 | 73.19% | 8 | 426.81 |

For large σ with small λ, expected removals **decreases** because the empirical cumulative gradient norms explode super-linearly — they grow much faster than the privacy budget (σε/c_δ) grows linearly in σ. A heavily perturbed, weakly-regularised model produces much larger Newton update steps per removal.

### How the Spikes Appear

`plot_accuracy_vs_removals` sorts all (σ, λ) points by E[removals] and connects them with lines. This interleaves points from very different σ values that happen to land at similar x-positions:

- (σ=100, E[rem]=8, acc=73%) sits next to (σ=0.03, E[rem]=6, acc=97%)
- (σ=30, E[rem]=28, acc=86%) sits next to (σ=0.3, E[rem]=15, acc=97%)

The line zig-zags between high-accuracy/low-σ points and low-accuracy/high-σ points, creating the dramatic downward spikes.

### Why the Original Paper Doesn't Show This

The paper almost certainly computes expected removals from the worst-case theoretical per-step bound (Theorem 1):

$$\beta_{\text{step}} = \frac{4\gamma C^2}{\lambda^2 (n-1)}$$

This bound is **independent of σ**, so E[T] = σε / (c_δ · β_step) is guaranteed monotonically increasing in σ. Our implementation uses actual empirical gradient norms from the removal run instead, which breaks this monotonicity for large σ with small λ.

---

## Figure 2 Differences from the Original Paper

Our initial Figure 2 differed from the paper's in several ways. This section
documents the causes and the fixes applied.

### Structural differences (resolved 2026-03-20)
1. **Missing "True value" line**: The paper plots the actual gradient residual
   norm ‖∇L(w_approx; D_rem)‖ after each Newton update. Now computed and
   plotted as a black dashed line.
2. **Missing batch bounds**: Worst-case batch (Theorem 3) and data-dependent
   batch (Corollary 2) were not plotted. Now included.
3. **Extra lines**: The original plot had a "per-step bound" line and a
   "budget" horizontal line that don't appear in the paper's Figure 2. Removed.

### Remaining numerical differences
1. **Random seed**: The perturbation vector b ~ N(0,σ²I) and removal
   permutation depend on the seed. Without the paper's seed, point-by-point
   agreement is impossible; only the overall scale and trend can match.
2. **Spectral norm approximation**: We use 20 iterations of power iteration
   vs the paper's unknown method. Small errors compound over 1000 steps.
3. **Theorem 1 constant**: Our per-step bound 4γC²/(λ²(n−1)) may differ
   from the paper's exact formula by a small constant, contributing to a
   ~2–3× vertical shift in the worst-case line.
4. **Theorem 3 / Corollary 2 formulas**: These are approximations inferred
   from the paper's theoretical statements. The exact expressions may differ.