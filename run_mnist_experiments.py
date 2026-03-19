#!/usr/bin/env python
"""
MNIST reproduction experiments for "Certified Data Removal" (Guo et al., ICML 2020).
Reproduces Section 4.1: Linear Logistic Regression on MNIST (digits 3 vs 8).

Usage:
    uv run python run_mnist_experiments.py                     # run all phases
    uv run python run_mnist_experiments.py --phase 2a          # accuracy grid only
    uv run python run_mnist_experiments.py --phase 2c          # removal grid
    uv run python run_mnist_experiments.py --phase 3           # 1000-removal trace
"""

import argparse
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time

PYTHON = sys.executable
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "save")
RESULT_DIR = os.path.join(PROJECT_DIR, "result")

# Paper constants (Section 4.1, Algorithm 2)
DELTA = 1e-4
C_DELTA = math.sqrt(2 * math.log(1.5 / DELTA))  # ≈ 4.39

# Hyperparameter grid from paper Figure 1
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1]
SIGMAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def run_removal(lam, std, num_removes, seed=42, num_steps=100):
    """Run test_removal.py and return parsed results."""
    cmd = [
        PYTHON, "test_removal.py",
        "--data-dir", DATA_DIR,
        "--result-dir", RESULT_DIR,
        "--dataset", "MNIST",
        "--extractor", "none",
        "--train-mode", "binary",
        "--lam", str(lam),
        "--std", str(std),
        "--num-removes", str(num_removes),
        "--num-steps", str(num_steps),
        "--seed", str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_DIR)

    if proc.returncode != 0:
        return {"error": proc.stderr.strip(), "returncode": proc.returncode,
                "lam": lam, "std": std, "seed": seed, "num_removes": num_removes}

    accs = re.findall(r'Test accuracy = ([\d.]+)', proc.stdout)
    train_time = re.search(r'Time elapsed: ([\d.]+)s', proc.stdout)
    norms = re.findall(r'Grad norm bound = ([\d.]+)', proc.stdout)
    rem_times = [float(t) for t in re.findall(r'time = ([\d.]+)s', proc.stdout)]

    return {
        "lam": lam, "std": std, "seed": seed,
        "num_removes": num_removes,
        "test_accuracy": float(accs[0]) if accs else None,
        "train_time_s": float(train_time.group(1)) if train_time else None,
        "grad_norms": [float(n) for n in norms],
        "removal_times_s": rem_times,
    }


def compute_expected_removals(grad_norms, std, epsilon=1.0):
    """Expected removals before the cumulative gradient norm budget is exhausted."""
    budget = std * epsilon / C_DELTA
    cumsum = 0.0
    for i, gn in enumerate(grad_norms):
        cumsum += gn
        if cumsum > budget:
            return i
    # All removals fit within budget — extrapolate from average
    if grad_norms and cumsum > 0:
        return int(budget / (cumsum / len(grad_norms)))
    return 0


# --------------------------------------------------------------------------- #
#  Phases                                                                      #
# --------------------------------------------------------------------------- #

def phase_2a():
    """Figure 1 left: test accuracy vs σ for each λ (no removal, training only)."""
    print("\n" + "=" * 70)
    print("  PHASE 2a — Test accuracy vs σ  (Figure 1, left panel)")
    print("=" * 70)

    results = []
    total = len(LAMBDAS) * len(SIGMAS)
    for i, (lam, std) in enumerate(itertools.product(LAMBDAS, SIGMAS)):
        tag = f"[{i+1}/{total}] λ={lam:.0e}, σ={std:.4g}"
        print(f"  {tag} ...", end=" ", flush=True)
        r = run_removal(lam, std, num_removes=0)
        acc = r.get("test_accuracy")
        t = r.get("train_time_s")
        status = f"acc={acc:.4f}" if acc else "FAILED"
        status += f"  ({t:.1f}s)" if t else ""
        print(status)
        results.append(r)

    # Summary table
    print("\n  Test Accuracy Grid")
    col_label = 'λ \\ σ'
    header = f"  {col_label:>10}" + "".join(f"{s:>9.4g}" for s in SIGMAS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lam in LAMBDAS:
        row = f"  {lam:>10.0e}"
        for std in SIGMAS:
            r = next((x for x in results if x["lam"] == lam and x["std"] == std), None)
            acc = r.get("test_accuracy") if r else None
            row += f"  {acc:>7.4f}" if acc else "     ERR"
        print(row)

    return results


def phase_2c(num_removes=100):
    """Figure 1 right + budget data: run removals across (λ, σ) grid."""
    print("\n" + "=" * 70)
    print(f"  PHASE 2c — Removal experiments ({num_removes} removals each)")
    print("=" * 70)

    results = []
    total = len(LAMBDAS) * len(SIGMAS)
    for i, (lam, std) in enumerate(itertools.product(LAMBDAS, SIGMAS)):
        tag = f"[{i+1}/{total}] λ={lam:.0e}, σ={std:.4g}"
        print(f"  {tag} ...", end=" ", flush=True)
        wall_start = time.time()
        r = run_removal(lam, std, num_removes=num_removes)
        wall = time.time() - wall_start
        acc = r.get("test_accuracy")

        if r.get("grad_norms"):
            er = compute_expected_removals(r["grad_norms"], std)
            r["expected_removals_eps1"] = er
            avg_t = sum(r["removal_times_s"]) / len(r["removal_times_s"])
            print(f"acc={acc:.4f}  E[rem]={er:>7d}  avg_step={avg_t:.3f}s  ({wall:.0f}s)")
        else:
            r["expected_removals_eps1"] = 0
            print(f"acc={acc:.4f}" if acc else "FAILED", f"({wall:.0f}s)")
        results.append(r)

    # Summary table
    print("\n  Expected Removals at ε=1 (δ=1e-4)")
    col_label = 'λ \\ σ'
    header = f"  {col_label:>10}" + "".join(f"{s:>9.4g}" for s in SIGMAS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lam in LAMBDAS:
        row = f"  {lam:>10.0e}"
        for std in SIGMAS:
            r = next((x for x in results if x["lam"] == lam and x["std"] == std), None)
            er = r.get("expected_removals_eps1", 0) if r else 0
            row += f"  {er:>7d}"
        print(row)

    return results


def phase_3(lam=1e-3, std=10.0, num_removes=1000):
    """Figure 2: gradient residual norm trace over 1000 removals."""
    print("\n" + "=" * 70)
    print(f"  PHASE 3 — 1000-removal trace  (λ={lam:.0e}, σ={std})")
    print("=" * 70)

    r = run_removal(lam, std, num_removes=num_removes)

    if not r.get("grad_norms"):
        print("  FAILED:", r.get("error", "unknown error"))
        return r

    norms = r["grad_norms"]
    cumsum = []
    s = 0.0
    for gn in norms:
        s += gn
        cumsum.append(s)

    budget = std * 1.0 / C_DELTA
    expected = compute_expected_removals(norms, std)

    # Worst-case bound (Theorem 1): γ=1/4, C=1 for logistic regression
    # Per single removal: 4γC²/(λ²(n-1))
    # Cumulative after T removals: T × per-removal
    n_train = 11982  # approx.  (3s + 8s in MNIST)
    worst_case_per = 4 * 0.25 * 1.0 / (lam**2 * (n_train - 1))

    print(f"  Test accuracy:              {r['test_accuracy']:.4f}")
    train_t = r.get('train_time_s') or 0
    print(f"  Train time:                 {train_t:.1f}s")
    print(f"  Total removal time:         {sum(r['removal_times_s']):.1f}s")
    print(f"  Avg step time:              {sum(r['removal_times_s'])/len(r['removal_times_s']):.3f}s")
    print(f"  Budget at ε=1:              {budget:.6f}")
    print(f"  Cumulative bound @1000:     {cumsum[-1]:.6f}")
    print(f"  Expected removals at ε=1:   {expected}")
    print(f"  Worst-case per removal:     {worst_case_per:.4e}")
    print(f"  Worst-case cumul @1000:     {worst_case_per * num_removes:.4e}")

    # Sample gradient norms at milestones
    milestones = [1, 10, 50, 100, 200, 500, 1000]
    print(f"\n  {'Step':>6}  {'Per-step bound':>14}  {'Cumul bound':>14}  {'Worst-case':>14}")
    print("  " + "-" * 54)
    for m in milestones:
        if m <= len(norms):
            print(f"  {m:>6d}  {norms[m-1]:>14.6f}  {cumsum[m-1]:>14.6f}  {worst_case_per * m:>14.4e}")

    r["cumulative_norms"] = cumsum
    r["expected_removals_eps1"] = expected
    r["worst_case_per_removal"] = worst_case_per
    return r


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="MNIST reproduction experiments")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "2a", "2c", "3"],
                        help="Phase to run (default: all)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--result-dir", type=str, default=None)
    args = parser.parse_args()

    global DATA_DIR, RESULT_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir
    if args.result_dir:
        RESULT_DIR = args.result_dir
    os.makedirs(RESULT_DIR, exist_ok=True)

    all_results = {}

    if args.phase in ("all", "2a"):
        all_results["phase_2a"] = phase_2a()

    if args.phase in ("all", "2c"):
        all_results["phase_2c"] = phase_2c(num_removes=100)

    if args.phase in ("all", "3"):
        all_results["phase_3"] = phase_3()

    # Persist structured results
    out_path = os.path.join(RESULT_DIR, "mnist_experiments.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
