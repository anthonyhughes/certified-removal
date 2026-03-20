# Experiment Plan

Reproduction of **Section 4.1** (Linear Logistic Regression on MNIST) from
*Certified Data Removal from Machine Learning Models* (Guo et al., ICML 2020).

## Dataset

| Property | Value |
|---|---|
| Dataset | MNIST — binary classification: digit 3 vs digit 8 |
| Training set | ~11,982 samples (~6,131 threes + ~5,851 eights) |
| Test set | ~1,984 samples |
| Features | 784-dim raw pixels, centered (x − 0.5), L2-normalised per sample |
| Labels | +1 (digit 3), −1 (digit 8) |

## Model & Algorithm

| Component | Detail |
|---|---|
| Model | L2-regularised binary logistic regression (Algorithm 1) |
| Optimiser | L-BFGS, 100 iterations |
| Loss perturbation | b ~ N(0, σ²I), added as b⊤w / n to the objective |
| Removal mechanism | Newton update (Algorithm 2): w⁻ = w* + H⁻¹Δ |
| Gradient residual bound | Data-dependent bound (Corollary 1) |
| Re-training trigger | Cumulative bound β > σε/c where c = √(2 log(1.5/δ)) |

## Hyperparameter Grid

From paper **Figure 1**:

| Parameter | Values |
|---|---|
| λ (L2 reg) | 1e-4, 1e-3, 1e-2, 1e-1 |
| σ (perturbation std) | 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0 |
| δ (failure prob) | 1e-4 (fixed) |
| ε (privacy param) | 1.0 (primary operating point) |
| Seed | 42 (fixed for single-seed runs) |

## Phases

### Phase 2a — Accuracy vs σ (Figure 1, left panel)
- **Goal**: Measure how test accuracy varies with σ for each λ.
- **Runs**: 4 λ × 9 σ = **36 runs**, `--num-removes 0` (train & evaluate only).
- **Output**: Test accuracy grid.
- **Command pattern**:
  ```
  uv run python test_removal.py --data-dir ./save --dataset MNIST \
    --extractor none --train-mode binary --lam <λ> --std <σ> \
    --num-removes 0 --num-steps 100 --seed 42 --result-dir ./result
  ```

### Phase 2c — Removal grid (Figure 1, right panel)
- **Goal**: For each (λ, σ), run 100 removals and compute the expected
  number of supported removals at ε = 1 before re-training is needed.
- **Runs**: 4 λ × 9 σ = **36 runs**, `--num-removes 100`.
- **Derived metric**: Expected removals = budget / avg per-step bound,
  where budget = σε/c.
- **Output**: Expected-removals grid; per-step gradient norm bounds.

### Phase 3 — Gradient residual trace (Figure 2)
- **Goal**: Track gradient residual norm bound over 1000 sequential removals.
- **Runs**: 1 run at λ = 1e-3, σ = 10.0, `--num-removes 1000`.
- **Comparisons**:
  - Data-dependent bound (Corollary 1) — from code.
  - Worst-case bound (Theorem 1): 4γC²/(λ²(n−1)) per removal (γ = 1/4, C = 1).
- **Output**: Per-step and cumulative gradient norm traces.

## Runner

All phases are orchestrated by `run_mnist_experiments.py`:
```
uv run python run_mnist_experiments.py              # all phases
uv run python run_mnist_experiments.py --phase 2a   # accuracy grid only
uv run python run_mnist_experiments.py --phase 3    # 1000-removal trace
```

Results are saved to `result/mnist_experiments.json`.

### Phase 3 extended comparisons (added 2026-03-20)
- **True gradient residual norm**: ‖∇L(w_approx; D_rem)‖ at each step.
- **Worst-case batch (Theorem 3)**: T·4γC²/(λ²(n−T)).
- **Data-dependent batch (Corollary 2)**: cumulative single bound × correction factor.

