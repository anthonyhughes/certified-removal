# Change History

## 2026-03-19 — Modernisation for local macOS reproduction

Original codebase from ICML 2020 paper: *Certified Data Removal from Machine Learning Models* (Guo et al.).
Updated to run locally on macOS (Intel x86_64) without a CUDA GPU.

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
