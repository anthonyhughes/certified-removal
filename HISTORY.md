# Change History

## 2026-03-19 / 9am — reproduction

Original codebase from ICML 2020 paper: *Certified Data Removal from Machine Learning Models* (Guo et al.).
Updated to run locally on a cpu.

### Scaffolding
- Added `pyproject.toml` with `uv` project management.
- Pinned dependencies: `torch==2.2.2`, `torchvision==0.17.2`, `opacus==1.4.0`, `numpy<2`, `scikit-learn`.
  - PyTorch 2.2.2 is the latest version with Intel macOS wheels.
  - `numpy<2` required for binary compatibility with torch 2.2.2.
- Created `save/` and `result/` output directories with `.gitkeep`.

### Dependency migration
- `train_svhn.py`: Replaced `torchdp.privacy_analysis` import with `opacus.accountants.analysis.rdp`. Updated `compute_rdp` and `get_privacy_spent` calls to use keyword-only arguments (new Opacus API).

### Device compatibility (CUDA → CPU/MPS)
- `train_svhn.py`, `test_removal.py`: Replaced hardcoded `torch.device("cuda")` with auto-detection (MPS if available, else CPU).
- `train_svhn.py`: Made `pin_memory` conditional on CUDA (no-op otherwise).
- `test_removal.py`: Changed `A.is_cuda` guard in `batch_multiply` to `A.device.type != 'cpu'`.

### Deprecated PyTorch API updates
- `test_removal.py`: `torch.ger` → `torch.outer` (3 sites).
- `test_removal.py`: `torch.autograd.Variable(...)` → `torch.zeros(...).requires_grad_(True)` (2 sites).
- `test_removal.py`: `.inverse()` → `torch.linalg.inv()`.
- `train_svhn.py`, `test_removal.py`, `utils.py`: Added `weights_only=True` to all `torch.load()` calls.

### Data loading
- `utils.py`: Added `download=True` to `datasets.MNIST()` calls so MNIST is auto-downloaded.

### Dry-run script
- Added `dry_run.py`: orchestrates a reduced end-to-end pipeline (1-epoch DP training → feature extraction → SVHN removal → MNIST 3-vs-8 removal) via subprocess. Reports pass/fail and timing per step. Uses a temp directory to avoid polluting `save/`/`result/`.

## 2026-03-19 / afternoon — MNIST reproduction experiments

### Reproducibility improvements
- `test_removal.py`: Added `--seed` argument (default 42). Seeds `torch.manual_seed` and `np.random.seed` before any stochastic operations.
- `test_removal.py`: Changed save-path format from `std_%.1f` to `std_%.4g` to prevent filename collisions for small σ values (e.g. σ = 0.01 and σ = 0.03 previously both truncated to `std_0.0`).

### Experiment runner
- Added `run_mnist_experiments.py`: orchestrates the full MNIST reproduction (Phases 2a, 2c, 3) via subprocess calls to `test_removal.py`. Parses stdout for accuracy and gradient norm bounds, computes expected number of supported removals from the cumulative budget, and saves structured results to `result/mnist_experiments.json`.

### Documentation
- Added `EXPERIMENTS.md`: describes the full experiment plan for reproducing Section 4.1 (hyperparameter grid, phases, commands).
- Added `RESULTS.md`: records all experimental results with tables and analysis.

## 2026-03-19 / eve — MNIST reproduction experiments

### Plotting
- Added `matplotlib` to project dependencies.
- Added `plot_results.py`: reads cached `.pth` results from `result/` and generates publication-style plots in `plots/`.
  - `fig1_left_accuracy_vs_sigma` — test accuracy vs σ for each λ (Figure 1 left).
  - `fig1_mid_accuracy_vs_epsilon` — test accuracy vs ε for 100 removals (Figure 1 middle).
  - `fig1_right_accuracy_vs_removals` — accuracy vs expected removals at ε=1 (Figure 1 right).
  - `fig1_right_annotated` — same as above with σ annotations on data points (paper style).
  - `fig2_gradient_residual_norms` — cumulative and per-step gradient norm bounds over 1000 removals (Figure 2).
  - `fig1_combined` — all four panels in a 2×2 grid figure.
  - Green colour palette matching the paper's visual style.
  - Each plot saved as both `.pdf` and `.png`.

## 2026-03-20 — Figure 2 alignment with paper

### test_removal.py
- Added computation of **true gradient residual norm** at each removal step: `‖∇L(w_approx; D_rem) + b‖ / n_rem`. Saved as `grad_norm_true` in the removal checkpoint alongside the existing `grad_norm_approx`.

### plot_results.py
- `plot_gradient_norms`: Replaced 3-line plot with full 5-line Figure 2 matching the paper:
  - Worst-case single (Theorem 1) — light blue
  - Worst-case batch (Theorem 3) — light green
  - Data-dependent single (Corollary 1) — blue
  - Data-dependent batch (Corollary 2) — green
  - True value (actual gradient residual norm) — black dashed
- Removed per-step bound and budget line (not in paper's Figure 2).
- `collect_data`: Now loads `grad_norm_true` from removal checkpoints.
- `plot_combined`: Updated Figure 2 panel with all 5 lines.

