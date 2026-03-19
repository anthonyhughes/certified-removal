#!/usr/bin/env python
"""
Dry-run script for Certified Data Removal.

Runs a minimal version of each pipeline step to verify that training,
feature extraction, and removal all work end-to-end on the current machine.

Steps:
  A) Train DP feature extractor on SVHN (1 epoch, train split only — no extra)
  B) Extract features from the trained model
  C) Run SVHN removal with 10 Newton-step removals
  D) Run MNIST 3-vs-8 binary removal with 10 removals (raw pixels, no extractor)

Usage:
  uv run python dry_run.py --svhn-dir <path-to-svhn> --mnist-dir <path-to-mnist>

If --svhn-dir is omitted, steps A-C are skipped.
If --mnist-dir is omitted, step D is skipped.
"""

import argparse
import os
import subprocess
import sys
import time
import tempfile
import shutil


def run_step(name, cmd, cwd):
    """Run a subprocess command and report pass/fail + timing."""
    print(f"\n{'='*60}")
    print(f"  STEP {name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n  >> STEP {name}: PASSED ({elapsed:.1f}s)")
    else:
        print(f"\n  >> STEP {name}: FAILED (exit code {result.returncode}, {elapsed:.1f}s)")
    return result.returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(description="Dry-run for certified removal pipeline")
    parser.add_argument("--svhn-dir", type=str, default=None,
                        help="Directory for SVHN data (will be downloaded if missing)")
    parser.add_argument("--mnist-dir", type=str, default=None,
                        help="Directory for MNIST data (will be downloaded if missing)")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable

    # Use a temporary save dir so we don't pollute the real save/result dirs
    tmp_dir = tempfile.mkdtemp(prefix="certified_removal_dry_run_")
    tmp_save = os.path.join(tmp_dir, "save")
    tmp_result = os.path.join(tmp_dir, "result")
    os.makedirs(tmp_save, exist_ok=True)
    os.makedirs(tmp_result, exist_ok=True)

    results = {}

    try:
        # --- Step A: Train DP extractor (1 epoch, SVHN train split only) ---
        if args.svhn_dir:
            passed, elapsed = run_step(
                "A — Train DP feature extractor (1 epoch)",
                [
                    python, "train_svhn.py",
                    "--data-dir", args.svhn_dir,
                    "--save-dir", tmp_save,
                    "--train-mode", "private",
                    "--std", "6",
                    "--delta", "1e-5",
                    "--normalize",
                    "--save-model",
                    "--epochs", "1",
                    "--batch-size", "200",
                    "--process-batch-size", "100",
                    "--log-interval", "5",
                ],
                cwd=project_dir,
            )
            results["A: DP Training"] = ("PASS" if passed else "FAIL", elapsed)

            if passed:
                # --- Step B: Extract features ---
                passed_b, elapsed_b = run_step(
                    "B — Extract features",
                    [
                        python, "train_svhn.py",
                        "--data-dir", args.svhn_dir,
                        "--save-dir", tmp_save,
                        "--test-mode", "extract",
                        "--std", "6",
                        "--delta", "1e-5",
                        "--normalize",
                    ],
                    cwd=project_dir,
                )
                results["B: Feature Extraction"] = ("PASS" if passed_b else "FAIL", elapsed_b)

                if passed_b:
                    # --- Step C: SVHN removal (10 removes) ---
                    passed_c, elapsed_c = run_step(
                        "C — SVHN certified removal (10 removes)",
                        [
                            python, "test_removal.py",
                            "--data-dir", args.svhn_dir,
                            "--result-dir", tmp_result,
                            "--extractor", "dp_delta_1.00e-05_std_6.00",
                            "--dataset", "SVHN",
                            "--std", "10",
                            "--lam", "2e-4",
                            "--num-steps", "20",
                            "--num-removes", "10",
                            "--subsample-ratio", "0.1",
                        ],
                        cwd=project_dir,
                    )
                    results["C: SVHN Removal"] = ("PASS" if passed_c else "FAIL", elapsed_c)
                else:
                    results["C: SVHN Removal"] = ("SKIPPED", 0)
            else:
                results["B: Feature Extraction"] = ("SKIPPED", 0)
                results["C: SVHN Removal"] = ("SKIPPED", 0)
        else:
            print("\n[INFO] --svhn-dir not provided, skipping steps A-C")
            results["A: DP Training"] = ("SKIPPED", 0)
            results["B: Feature Extraction"] = ("SKIPPED", 0)
            results["C: SVHN Removal"] = ("SKIPPED", 0)

        # --- Step D: MNIST 3-vs-8 removal (10 removes, raw pixels) ---
        if args.mnist_dir:
            passed_d, elapsed_d = run_step(
                "D — MNIST 3-vs-8 binary removal (10 removes)",
                [
                    python, "test_removal.py",
                    "--data-dir", args.mnist_dir,
                    "--result-dir", tmp_result,
                    "--extractor", "none",
                    "--dataset", "MNIST",
                    "--train-mode", "binary",
                    "--std", "10",
                    "--lam", "1e-3",
                    "--num-steps", "20",
                    "--num-removes", "10",
                ],
                cwd=project_dir,
            )
            results["D: MNIST 3v8 Removal"] = ("PASS" if passed_d else "FAIL", elapsed_d)
        else:
            print("\n[INFO] --mnist-dir not provided, skipping step D")
            results["D: MNIST 3v8 Removal"] = ("SKIPPED", 0)

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  DRY RUN SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for step, (status, elapsed) in results.items():
        marker = {"PASS": "+", "FAIL": "X", "SKIPPED": "-"}[status]
        time_str = f"({elapsed:.1f}s)" if elapsed > 0 else ""
        print(f"  [{marker}] {step}: {status} {time_str}")
        if status == "FAIL":
            all_passed = False
    print(f"{'='*60}")

    if not all_passed:
        print("\nSome steps FAILED. Check the output above for details.")
        sys.exit(1)
    else:
        print("\nAll executed steps passed!")


if __name__ == "__main__":
    main()
