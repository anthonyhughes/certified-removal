#!/usr/bin/env python
"""
Generate publication-style plots for the MNIST reproduction experiments.

Reads cached results from result/ and produces 4 plots in plots/:
  1. fig1_left_accuracy_vs_sigma.pdf   — Test accuracy vs σ
  2. fig1_right_accuracy_vs_removals.pdf — Accuracy vs expected removals (ε=1)
  3. fig2_gradient_residual_norms.pdf  — Gradient norm bounds over 1000 removals
  4. fig1_combined.pdf                 — 3-panel combined figure

Usage:
    uv run python plot_results.py
"""

import itertools
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #

RESULT_DIR = os.path.join(os.path.dirname(__file__), "result")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1]
SIGMAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

DELTA = 1e-4
C_DELTA = math.sqrt(2 * math.log(1.5 / DELTA))  # ≈ 4.39
GAMMA = 0.25   # Lipschitz constant of ℓ'' for logistic regression
C_GRAD = 1.0   # gradient norm bound for logistic loss with ‖x‖≤1

LAM_COLOURS = {1e-4: "#1b6d35", 1e-3: "#3da660", 1e-2: "#8dc99a", 1e-1: "#c7e3cc"}
LAM_LABELS = {1e-4: r"$\lambda=10^{-4}$", 1e-3: r"$\lambda=10^{-3}$",
              1e-2: r"$\lambda=10^{-2}$", 1e-1: r"$\lambda=10^{-1}$"}

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# --------------------------------------------------------------------------- #
#  Data loading                                                                #
# --------------------------------------------------------------------------- #

def load_model(lam, std):
    """Load cached training result (w, b) and compute test accuracy."""
    path = os.path.join(RESULT_DIR,
        f"none_MNIST_splits_1_ratio_1.00_std_{std:.4g}_lam_{lam:.0e}.pth")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    return ckpt


def load_removal(lam, std):
    """Load cached removal result (grad_norm_approx, times)."""
    path = os.path.join(RESULT_DIR,
        f"none_MNIST_splits_1_ratio_1.00_std_{std:.4g}_lam_{lam:.0e}_removal.pth")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    return ckpt


def get_test_accuracy(lam, std):
    """Compute test accuracy from cached model weights."""
    from torchvision import datasets, transforms
    data_dir = os.path.join(os.path.dirname(__file__), "save")

    ckpt = load_model(lam, std)
    if ckpt is None:
        return None
    w = ckpt["w"]

    # Load MNIST 3 vs 8 test set (same as utils.load_features)
    testset = datasets.MNIST(data_dir, train=False, download=True,
                             transform=transforms.ToTensor())
    X_test = torch.zeros(len(testset), 784)
    y_test = torch.zeros(len(testset))
    for i in range(len(testset)):
        x, y = testset[i]
        X_test[i] = x.view(784) - 0.5
        y_test[i] = y
    mask = (y_test.eq(3) + y_test.eq(8)).gt(0)
    X_test = X_test[mask]
    y_test = y_test[mask].eq(3).float()  # 1 = digit 3, 0 = digit 8
    X_test /= X_test.norm(2, 1).unsqueeze(1)

    pred = X_test.mv(w)
    acc = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
    return acc


def compute_n_train():
    """Return the number of MNIST 3-vs-8 training samples."""
    from torchvision import datasets, transforms
    data_dir = os.path.join(os.path.dirname(__file__), "save")
    trainset = datasets.MNIST(data_dir, train=True, download=True,
                              transform=transforms.ToTensor())
    y = torch.tensor([trainset[i][1] for i in range(len(trainset))])
    return int((y.eq(3) | y.eq(8)).sum().item())


def compute_expected_removals(grad_norms, std, epsilon=1.0):
    """Expected removals before cumulative bound exceeds budget."""
    budget = std * epsilon / C_DELTA
    cumsum = 0.0
    for i, gn in enumerate(grad_norms):
        cumsum += float(gn)
        if cumsum > budget:
            return i
    if len(grad_norms) > 0 and cumsum > 0:
        return int(budget / (cumsum / len(grad_norms)))
    return 0


def compute_expected_removals_worstcase(lam, std, n_train, epsilon=1.0):
    """Expected removals using the worst-case per-step bound (Theorem 1)."""
    per_step = 4 * GAMMA * C_GRAD**2 / (lam**2 * (n_train - 1))
    budget = std * epsilon / C_DELTA
    if per_step <= 0:
        return 0
    return int(budget / per_step)

# --------------------------------------------------------------------------- #
#  Collect all data                                                            #
# --------------------------------------------------------------------------- #

def collect_data():
    """Build accuracy and removal grids from cached results."""
    acc_grid = {}     # (lam, std) -> accuracy
    er_grid = {}      # (lam, std) -> expected_removals (data-dependent)
    er_wc_grid = {}   # (lam, std) -> expected_removals (worst-case Thm. 1)
    norms_grid = {}   # (lam, std) -> list of per-step gradient norms

    n_train = compute_n_train()

    for lam, std in itertools.product(LAMBDAS, SIGMAS):
        acc = get_test_accuracy(lam, std)
        if acc is not None:
            acc_grid[(lam, std)] = acc
            er_wc_grid[(lam, std)] = compute_expected_removals_worstcase(
                lam, std, n_train)

        rem = load_removal(lam, std)
        if rem is not None:
            gn = rem["grad_norm_approx"].tolist()
            norms_grid[(lam, std)] = gn
            er_grid[(lam, std)] = compute_expected_removals(gn, std)

    return acc_grid, er_grid, er_wc_grid, norms_grid, n_train


# --------------------------------------------------------------------------- #
#  Plot 0: Accuracy vs ε for 100 removals  (Figure 1, middle)                 #
# --------------------------------------------------------------------------- #

def plot_accuracy_vs_epsilon(acc_grid, norms_grid, ax=None, num_removals=100,
                             n_train=None):
    """For each (λ, σ), compute the minimum ε that supports `num_removals`
    removals: ε = c · β_T / σ, where β_T is cumulative bound after T removals.
    Then plot accuracy vs ε for each λ.

    When n_train is provided, also overlay worst-case (Thm. 1) curves as
    dashed lines."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 3.8))

    for lam in LAMBDAS:
        # --- data-dependent (Corollary 1) ---
        points = []  # (epsilon, accuracy)
        for std in SIGMAS:
            if (lam, std) not in acc_grid or (lam, std) not in norms_grid:
                continue
            gn = norms_grid[(lam, std)]
            if len(gn) < num_removals:
                continue
            beta_T = sum(gn[:num_removals])
            if beta_T <= 0:
                continue
            eps = C_DELTA * beta_T / std
            acc = acc_grid[(lam, std)] * 100
            points.append((eps, acc))
        if points:
            points.sort()
            epsilons = [p[0] for p in points]
            accs = [p[1] for p in points]
            ax.plot(epsilons, accs, "o-", color=LAM_COLOURS[lam],
                    label=LAM_LABELS[lam], markersize=4, linewidth=1.5)

        # --- worst-case (Theorem 1) ---
        if n_train is not None:
            wc_points = []
            per_step = 4 * GAMMA * C_GRAD**2 / (lam**2 * (n_train - 1))
            beta_T_wc = num_removals * per_step
            for std in SIGMAS:
                if (lam, std) not in acc_grid:
                    continue
                eps_wc = C_DELTA * beta_T_wc / std
                acc = acc_grid[(lam, std)] * 100
                wc_points.append((eps_wc, acc))
            if wc_points:
                wc_points.sort()
                ax.plot([p[0] for p in wc_points], [p[1] for p in wc_points],
                        "s--", color=LAM_COLOURS[lam], markersize=3,
                        linewidth=1.2, alpha=0.7)

    # Legend: add a single entry for the dashed style
    if n_train is not None:
        ax.plot([], [], "s--", color="grey", markersize=3, linewidth=1.2,
                alpha=0.7, label="Worst-case (Thm. 1)")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Accuracy vs $\\varepsilon$ (supporting {num_removals} removals)")
    ax.set_ylim(72, 98)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if standalone:
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "fig1_mid_accuracy_vs_epsilon.pdf"))
        fig.savefig(os.path.join(PLOT_DIR, "fig1_mid_accuracy_vs_epsilon.png"))
        plt.close(fig)
        print("  Saved fig1_mid_accuracy_vs_epsilon.{pdf,png}")


# --------------------------------------------------------------------------- #
#  Plot 1: Accuracy vs σ  (Figure 1, left)                                    #
# --------------------------------------------------------------------------- #

def plot_accuracy_vs_sigma(acc_grid, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 3.8))

    for lam in LAMBDAS:
        sigs = [s for s in SIGMAS if (lam, s) in acc_grid]
        accs = [acc_grid[(lam, s)] * 100 for s in sigs]
        ax.plot(sigs, accs, "o-", color=LAM_COLOURS[lam], label=LAM_LABELS[lam],
                markersize=4, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Effect of $\\lambda$ and $\\sigma$ on Accuracy")
    ax.set_ylim(65, 100)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    if standalone:
        fig.savefig(os.path.join(PLOT_DIR, "fig1_left_accuracy_vs_sigma.pdf"))
        fig.savefig(os.path.join(PLOT_DIR, "fig1_left_accuracy_vs_sigma.png"))
        plt.close(fig)
        print("  Saved fig1_left_accuracy_vs_sigma.{pdf,png}")


# --------------------------------------------------------------------------- #
#  Plot 2: Accuracy vs expected removals  (Figure 1, right)                   #
# --------------------------------------------------------------------------- #

def plot_accuracy_vs_removals(acc_grid, er_grid, ax=None, er_wc_grid=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 3.8))

    for lam in LAMBDAS:
        # --- data-dependent (Corollary 1) ---
        points = []
        for std in SIGMAS:
            if (lam, std) in acc_grid and (lam, std) in er_grid:
                acc = acc_grid[(lam, std)] * 100
                er = er_grid[(lam, std)]
                if er > 0:
                    points.append((er, acc, std))
        if points:
            points.sort()
            ers = [p[0] for p in points]
            accs = [p[1] for p in points]
            ax.plot(ers, accs, "o-", color=LAM_COLOURS[lam], label=LAM_LABELS[lam],
                    markersize=4, linewidth=1.5)

        # --- worst-case (Theorem 1) ---
        if er_wc_grid is not None:
            wc_points = []
            for std in SIGMAS:
                if (lam, std) in acc_grid and (lam, std) in er_wc_grid:
                    acc = acc_grid[(lam, std)] * 100
                    er_wc = er_wc_grid[(lam, std)]
                    if er_wc > 0:
                        wc_points.append((er_wc, acc))
            if wc_points:
                wc_points.sort()
                ax.plot([p[0] for p in wc_points], [p[1] for p in wc_points],
                        "s--", color=LAM_COLOURS[lam], markersize=3,
                        linewidth=1.2, alpha=0.7)

    if er_wc_grid is not None:
        ax.plot([], [], "s--", color="grey", markersize=3, linewidth=1.2,
                alpha=0.7, label="Worst-case (Thm. 1)")

    ax.set_xscale("log")
    ax.set_xlabel("Expected # of Supported Removals")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs Removals ($\\varepsilon=1$)")
    ax.set_ylim(65, 100)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    if standalone:
        fig.savefig(os.path.join(PLOT_DIR, "fig1_right_accuracy_vs_removals.pdf"))
        fig.savefig(os.path.join(PLOT_DIR, "fig1_right_accuracy_vs_removals.png"))
        plt.close(fig)
        print("  Saved fig1_right_accuracy_vs_removals.{pdf,png}")


# --------------------------------------------------------------------------- #
#  Plot 2b: Accuracy vs removals with σ annotations (paper style)             #
# --------------------------------------------------------------------------- #

# σ values to annotate (subset for readability)
_ANNOTATE_SIGMAS = {0.01, 0.1, 1.0, 10.0, 100.0}

def plot_accuracy_vs_removals_annotated(acc_grid, er_grid, ax=None,
                                        er_wc_grid=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))

    for lam in LAMBDAS:
        # --- data-dependent (Corollary 1) ---
        points = []
        for std in SIGMAS:
            if (lam, std) in acc_grid and (lam, std) in er_grid:
                acc = acc_grid[(lam, std)] * 100
                er = er_grid[(lam, std)]
                if er > 0:
                    points.append((er, acc, std))
        if not points:
            continue
        points.sort()
        ers = [p[0] for p in points]
        accs = [p[1] for p in points]
        ax.plot(ers, accs, "o-", color=LAM_COLOURS[lam], label=LAM_LABELS[lam],
                markersize=5, linewidth=1.8)

        # Annotate selected σ values (only on top curve to avoid clutter,
        # but annotate all curves for the first/last sigma)
        for er_val, acc_val, sig_val in points:
            if sig_val in _ANNOTATE_SIGMAS:
                if lam == 1e-4 or sig_val in (0.01, 100.0):
                    if sig_val <= 0.1:
                        offset = (8, 8)
                    elif sig_val >= 100:
                        offset = (8, -12)
                    else:
                        offset = (8, 6)
                    ax.annotate(
                        f"$\\sigma={sig_val:g}$",
                        xy=(er_val, acc_val), fontsize=7.5,
                        textcoords="offset points", xytext=offset,
                        color=LAM_COLOURS[lam], alpha=0.85,
                    )

        # --- worst-case (Theorem 1) ---
        if er_wc_grid is not None:
            wc_points = []
            for std in SIGMAS:
                if (lam, std) in acc_grid and (lam, std) in er_wc_grid:
                    acc = acc_grid[(lam, std)] * 100
                    er_wc = er_wc_grid[(lam, std)]
                    if er_wc > 0:
                        wc_points.append((er_wc, acc, std))
            if wc_points:
                wc_points.sort()
                ax.plot([p[0] for p in wc_points], [p[1] for p in wc_points],
                        "s--", color=LAM_COLOURS[lam], markersize=3,
                        linewidth=1.2, alpha=0.7)

    if er_wc_grid is not None:
        ax.plot([], [], "s--", color="grey", markersize=3, linewidth=1.2,
                alpha=0.7, label="Worst-case (Thm. 1)")

    ax.set_xscale("log")
    ax.set_xlabel("Expected # of Supported Removals")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs Removals ($\\varepsilon=1$)")
    ax.set_ylim(72, 98)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    if standalone:
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "fig1_right_annotated.pdf"))
        fig.savefig(os.path.join(PLOT_DIR, "fig1_right_annotated.png"))
        plt.close(fig)
        print("  Saved fig1_right_annotated.{pdf,png}")


# --------------------------------------------------------------------------- #
#  Plot 3: Gradient residual norms  (Figure 2)                                #
# --------------------------------------------------------------------------- #

def plot_gradient_norms(norms_grid):
    lam, std = 1e-3, 10.0
    if (lam, std) not in norms_grid:
        print("  SKIP fig2: no 1000-removal data for λ=1e-3, σ=10")
        return

    norms = norms_grid[(lam, std)]
    n_removals = len(norms)
    xs = np.arange(1, n_removals + 1)

    # Cumulative data-dependent bound
    cumul = np.cumsum(norms)

    # Worst-case bound (Theorem 1)
    n_train = 11982  # approximate MNIST 3v8 size
    worst_per = 4 * GAMMA * C_GRAD**2 / (lam**2 * (n_train - 1))
    worst_cumul = worst_per * xs

    fig, ax = plt.subplots(figsize=(6, 4.2))

    ax.semilogy(xs, cumul, "-", color="#1f77b4", linewidth=1.5,
                label="Data-dependent (Corollary 1)")
    ax.semilogy(xs, worst_cumul, "-", color="#aec7e8", linewidth=1.5,
                label="Worst-case (Theorem 1)")
    ax.semilogy(xs, norms, "-", color="#ff7f0e", linewidth=0.8, alpha=0.7,
                label="Per-step bound")

    # Budget line
    budget = std * 1.0 / C_DELTA
    ax.axhline(budget, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Budget $\\sigma\\varepsilon/c = {budget:.2f}$")

    ax.set_xlabel("# of Removals")
    ax.set_ylabel("Gradient Residual Norm")
    ax.set_title(f"Gradient Residual Norm Bounds ($\\lambda={lam:.0e}$, $\\sigma={std:.0f}$)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_xlim(0, n_removals)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(PLOT_DIR, "fig2_gradient_residual_norms.pdf"))
    fig.savefig(os.path.join(PLOT_DIR, "fig2_gradient_residual_norms.png"))
    plt.close(fig)
    print("  Saved fig2_gradient_residual_norms.{pdf,png}")


# --------------------------------------------------------------------------- #
#  Combined 3-panel figure                                                     #
# --------------------------------------------------------------------------- #

def plot_combined(acc_grid, er_grid, norms_grid, er_wc_grid=None, n_train=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) Accuracy vs σ
    plot_accuracy_vs_sigma(acc_grid, ax=axes[0, 0])

    # (0,1) Accuracy vs ε for 100 removals
    plot_accuracy_vs_epsilon(acc_grid, norms_grid, ax=axes[0, 1],
                             n_train=n_train)

    # (1,0) Accuracy vs removals (annotated with σ labels)
    plot_accuracy_vs_removals_annotated(acc_grid, er_grid, ax=axes[1, 0],
                                        er_wc_grid=er_wc_grid)

    # (1,1) Gradient residual norms
    lam, std = 1e-3, 10.0
    if (lam, std) in norms_grid:
        norms = norms_grid[(lam, std)]
        n_removals = len(norms)
        xs = np.arange(1, n_removals + 1)
        cumul = np.cumsum(norms)
        _n = n_train if n_train is not None else 11982
        worst_per = 4 * GAMMA * C_GRAD**2 / (lam**2 * (_n - 1))
        worst_cumul = worst_per * xs
        budget = std * 1.0 / C_DELTA

        ax = axes[1, 1]
        ax.semilogy(xs, cumul, "-", color="#1b6d35", linewidth=1.5,
                    label="Data-dep. (Cor. 1)")
        ax.semilogy(xs, worst_cumul, "-", color="#c7e3cc", linewidth=1.5,
                    label="Worst-case (Thm. 1)")
        ax.semilogy(xs, norms, "-", color="#3da660", linewidth=0.8, alpha=0.7,
                    label="Per-step bound")
        ax.axhline(budget, color="#d62728", linestyle="--", linewidth=1, alpha=0.7,
                   label=f"Budget ($\\sigma\\varepsilon/c$)")
        ax.set_xlabel("# of Removals")
        ax.set_ylabel("Gradient Residual Norm")
        ax.set_title(f"Grad. Norm Bounds ($\\lambda=10^{{-3}}$, $\\sigma=10$)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(0, n_removals)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Reproduction of Guo et al. (ICML 2020) — MNIST 3 vs 8",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig1_combined.pdf"))
    fig.savefig(os.path.join(PLOT_DIR, "fig1_combined.png"))
    plt.close(fig)
    print("  Saved fig1_combined.{pdf,png}")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    print("Collecting data from result/ ...")
    acc_grid, er_grid, er_wc_grid, norms_grid, n_train = collect_data()
    print(f"  {len(acc_grid)} accuracy entries, {len(er_grid)} removal entries, "
          f"{len(norms_grid)} norm traces, n_train={n_train}\n")

    print("Generating plots ...")
    plot_accuracy_vs_sigma(acc_grid)
    plot_accuracy_vs_epsilon(acc_grid, norms_grid, n_train=n_train)
    plot_accuracy_vs_removals(acc_grid, er_grid, er_wc_grid=er_wc_grid)
    plot_accuracy_vs_removals_annotated(acc_grid, er_grid, er_wc_grid=er_wc_grid)
    plot_gradient_norms(norms_grid)
    plot_combined(acc_grid, er_grid, norms_grid,
                  er_wc_grid=er_wc_grid, n_train=n_train)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
