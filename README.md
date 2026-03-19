# Certified Data Removal from Machine Learning Models

### Dependencies

Managed via [uv](https://docs.astral.sh/uv/). Core packages: torch, torchvision, scikit-learn, opacus.

### Setup

```bash
uv sync
```

This creates a virtual environment and installs all pinned dependencies. The `save/` and `result/` directories are included in the repo.

### Dry run

Verify the full pipeline works end-to-end with reduced data before committing to a long training run:

```bash
# MNIST only (fast, ~30s, downloads ~60MB)
uv run python dry_run.py --mnist-dir ./data/mnist

# Full pipeline including SVHN (downloads ~2.4GB)
uv run python dry_run.py --svhn-dir ./data/svhn --mnist-dir ./data/mnist
```

This trains for 1 epoch with small batches, extracts features, and runs 10 Newton-step removals — reporting pass/fail and timing for each step.

### Training a differentially private (DP) feature extractor

Training a (0.1, 1e-5)-differentially private feature extractor for SVHN:

```bash
uv run python train_svhn.py --data-dir <SVHN path> --train-mode private --std 6 --delta 1e-5 --normalize --save-model
```

Extracting features using the differentially private extractor:

```bash
uv run python train_svhn.py --data-dir <SVHN path> --test-mode extract --std 6 --delta 1e-5
```

### Removing data from trained model

Training a removal-enabled one-vs-all linear classifier and removing 1000 training points:

```bash
uv run python test_removal.py --data-dir <SVHN path> --verbose --extractor dp_delta_1.00e-05_std_6.00 --dataset SVHN --std 10 --lam 2e-4 --num-steps 100 --subsample-ratio 0.1
```

This script randomly samples 1000 training points and applies the Newton update removal mechanism.
The total gradient residual norm bound is accumulated, which governs how many of the 1000 training points can be removed before re-training.
For this setting, the number of certifiably removed training points is limited by the DP feature extractor.

### Removing data from an MNIST 3 vs. 8 model

Training a removal-enabled binary logistic regression classifier for MNIST 3 vs. 8 and removing 1000 training points:

```bash
uv run python test_removal.py --data-dir <MNIST path> --verbose --extractor none --dataset MNIST --train-mode binary --std 10 --lam 1e-3 --num-steps 100
```

### Reference

This code corresponds to the following paper:

Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens van der Maaten. **[Certified Data Removal from Machine Learning Models](https://arxiv.org/pdf/1911.03030.pdf)**. ICML 2020.

### Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

### License
This project is CC-BY-NC 4.0 licensed, as found in the LICENSE file.
