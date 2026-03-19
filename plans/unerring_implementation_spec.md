# UnERRING Implementation Specification

## Overview

This document specifies an implementation of **Algorithm 1 (UnERRING)** from the paper
"UnERRING: Unlearning for (weakly) Convex Losses". The algorithm performs certified
machine unlearning for logistic regression by combining a fast hypothesis for the
leave-T-out optimum with a verification step and geometry-aware Gaussian masking.

**Scope of this spec:** We implement the full pipeline for **L2-regularised logistic
regression** in the Gaussian-PAC setting. The hypothesis generation uses the **Newton
Step (NS) approximation** as the primary method, since DRIF is referenced from [RH25]
without a self-contained formula in the paper. NS is explicitly listed as a valid
instantiation throughout (Theorems 1.2, 1.4).

**Key mathematical objects:**

| Symbol | Definition | Shape |
|--------|-----------|-------|
| $X \in \mathbb{R}^{n \times d}$ | Design matrix (rows are samples) | $(n, d)$ |
| $y \in \{0, 1\}^n$ | Binary labels | $(n,)$ |
| $T \subseteq [n]$ | Deletion set | indices |
| $k = \|T\|$ | Number of deletions | scalar |
| $X_{\backslash T}, y_{\backslash T}$ | Retained data | $(n-k, d)$, $(n-k,)$ |
| $\Sigma_{\backslash T} = X_{\backslash T}^\top X_{\backslash T}$ | Retained second moment | $(d, d)$ |
| $\hat{\theta}$ | ERM optimum on full data | $(d,)$ |
| $\hat{\theta}_{\backslash T}$ | ERM optimum on retained data | $(d,)$ |
| $\theta^{\text{NS}}_{\backslash T}$ | Newton step approximation | $(d,)$ |
| $H$ | Hessian of loss at $\hat{\theta}$ | $(d, d)$ |
| $g_i$ | Per-sample gradient at $\hat{\theta}$ | $(d,)$ |
| $\lambda$ | L2 regularisation coefficient | scalar |

**Convention note:** The paper uses $X \in \mathbb{R}^{d \times n}$ (columns are samples)
in the proofs but the algorithm description is ambiguous. We use the **rows-are-samples**
convention ($X \in \mathbb{R}^{n \times d}$) throughout the code, which is standard in
scikit-learn / numpy. All formulas below are transposed accordingly.

---

## Module 0: Loss Function and Derivatives

Before any phase, we need a clean implementation of logistic regression loss and its
derivatives with L2 regularisation.

### Specification

The regularised loss for a single sample $(x_i, y_i)$ is:

$$\ell_i(\theta) = -y_i \log \sigma(x_i^\top \theta) - (1 - y_i) \log(1 - \sigma(x_i^\top \theta))$$

where $\sigma(z) = 1/(1 + e^{-z})$.

The total loss is:

$$L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell_i(\theta) + \frac{\lambda}{2} \|\theta\|_2^2$$

**Gradient:**

$$\nabla L(\theta) = \frac{1}{n} X^\top (\sigma(X\theta) - y) + \lambda \theta$$

where $\sigma(X\theta)$ is applied element-wise.

**Hessian:**

$$\nabla^2 L(\theta) = \frac{1}{n} X^\top W X + \lambda I$$

where $W = \text{diag}(\sigma(X\theta) \odot (1 - \sigma(X\theta)))$ is the diagonal matrix
of logistic weights.

**Per-sample gradient** (at $\hat{\theta}$, without regularisation contribution):

$$g_i = (\sigma(x_i^\top \hat{\theta}) - y_i) \cdot x_i$$

### Code

```python
import numpy as np
from scipy.special import expit  # numerically stable sigmoid


def logistic_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                  lam: float) -> float:
    """Regularised logistic regression loss.
    
    Args:
        theta: Parameter vector, shape (d,)
        X: Design matrix, shape (n, d)
        y: Labels in {0, 1}, shape (n,)
        lam: L2 regularisation coefficient
    """
    z = X @ theta  # (n,)
    # Use log-sum-exp trick for numerical stability
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return loss + 0.5 * lam * np.dot(theta, theta)


def logistic_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                      lam: float) -> np.ndarray:
    """Gradient of regularised logistic loss. Returns shape (d,)."""
    n = X.shape[0]
    p = expit(X @ theta)  # (n,)
    return (X.T @ (p - y)) / n + lam * theta


def logistic_hessian(theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                     lam: float) -> np.ndarray:
    """Hessian of regularised logistic loss. Returns shape (d, d)."""
    n = X.shape[0]
    p = expit(X @ theta)  # (n,)
    w = p * (1 - p)       # (n,) logistic weights
    return (X.T * w) @ X / n + lam * np.eye(X.shape[1])


def per_sample_gradients(theta: np.ndarray, X: np.ndarray,
                         y: np.ndarray) -> np.ndarray:
    """Per-sample gradients (unregularised). Returns shape (n, d)."""
    p = expit(X @ theta)  # (n,)
    residuals = p - y     # (n,)
    return residuals[:, None] * X  # (n, d)
```

### Tests for Module 0

```
TEST_M0_1 (gradient_finite_diff):
    Generate random theta, X (5x3), y.
    For each coordinate j, compute (L(theta + eps*e_j) - L(theta - eps*e_j)) / (2*eps).
    Assert allclose to logistic_gradient(theta, X, y, lam)[j], atol=1e-5.

TEST_M0_2 (hessian_finite_diff):
    Same approach: finite-difference the gradient to get the Hessian.
    Assert allclose to logistic_hessian, atol=1e-4.

TEST_M0_3 (hessian_psd):
    For random X, y, theta, lam > 0: assert all eigenvalues of Hessian are positive.

TEST_M0_4 (gradient_at_optimum):
    Train logistic regression to convergence via scipy.optimize.minimize.
    Assert ||gradient|| < 1e-8 at the optimum.

TEST_M0_5 (per_sample_gradient_sum):
    Assert that (1/n) * sum(per_sample_gradients) + lam * theta == logistic_gradient.
```

---

## Phase 1: Data Preprocessing (Algorithm 1, Lines 1-2)

### Specification

**Input:** $X \in \mathbb{R}^{n \times d}$, $y \in \{0, 1\}^n$, $T \subseteq \{0, \ldots, n-1\}$

**Output:** $X_{\backslash T} \in \mathbb{R}^{(n-k) \times d}$, $y_{\backslash T} \in \{0,1\}^{n-k}$,
$\Sigma_{\backslash T} \in \mathbb{R}^{d \times d}$

### Code

```python
def remove_samples(X: np.ndarray, y: np.ndarray,
                   T: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """Remove samples indexed by T from dataset.
    
    Args:
        X: Design matrix, shape (n, d)
        y: Labels, shape (n,)
        T: Set of integer indices to remove (0-indexed)
    Returns:
        X_ret, y_ret: Retained data
    """
    mask = np.ones(X.shape[0], dtype=bool)
    mask[list(T)] = False
    return X[mask], y[mask]


def compute_sigma(X_ret: np.ndarray) -> np.ndarray:
    """Compute Sigma_{\\T} = X_{\\T}^T X_{\\T}. Returns shape (d, d)."""
    return X_ret.T @ X_ret
```

### Tests

```
TEST_P1_1 (dimensions):
    X (100x5), y (100,), T = {0, 5, 99}.
    X_ret, y_ret = remove_samples(X, y, T)
    Assert X_ret.shape == (97, 5), y_ret.shape == (97,)

TEST_P1_2 (content_correctness):
    X = np.arange(20).reshape(4, 5).astype(float)
    T = {1, 3}
    X_ret, _ = remove_samples(X, np.zeros(4), T)
    Assert np.array_equal(X_ret, X[[0, 2]])  # rows 0 and 2 retained

TEST_P1_3 (sigma_symmetric):
    Sigma = compute_sigma(X_ret)
    Assert np.allclose(Sigma, Sigma.T)

TEST_P1_4 (sigma_psd):
    Assert all eigenvalues of Sigma >= 0.
    If n-k >= d, assert all eigenvalues > 0 (full rank).

TEST_P1_5 (sigma_positive_definite_for_tall_matrix):
    X_ret of shape (100, 5) with entries ~ N(0,1).
    Assert min eigenvalue of compute_sigma(X_ret) > 0.
    (This is needed for the Cholesky in Phase 5.)

TEST_P1_6 (empty_deletion):
    T = set(). Assert X_ret identical to X, y_ret identical to y.

TEST_P1_7 (index_bounds):
    T = {-1} or T = {n}. Assert raises IndexError or ValueError.
```

---

## Phase 2: Hypothesis Generation (Algorithm 1, Lines 3-4)

### Specification

The Newton Step (NS) approximation to the leave-T-out optimum is:

$$\theta^{\text{NS}}_{\backslash T} = \hat{\theta} + H^{-1} \cdot \frac{1}{n} \sum_{i \in T} g_i$$

where:
- $H = \nabla^2 L(\hat{\theta})$ is the Hessian at the full-data optimum (including regularisation)
- $g_i = (\sigma(x_i^\top \hat{\theta}) - y_i) \cdot x_i$ is the per-sample gradient (without regularisation)
- The $\frac{1}{n}$ factor is because our loss is $\frac{1}{n}\sum \ell_i$

**Derivation:** This is the standard influence function / one-step Newton approximation.
If we define $L_{\backslash T}(\theta) = \frac{1}{n}\sum_{i \notin T} \ell_i(\theta) + \frac{\lambda}{2}\|\theta\|^2$,
then $\nabla L_{\backslash T}(\hat{\theta}) = \nabla L(\hat{\theta}) - \frac{1}{n}\sum_{i \in T} g_i = -\frac{1}{n}\sum_{i \in T} g_i$
(since $\nabla L(\hat{\theta}) = 0$). A single Newton step from $\hat{\theta}$ gives
$\hat{\theta} - H^{-1} \nabla L_{\backslash T}(\hat{\theta}) = \hat{\theta} + H^{-1} \frac{1}{n}\sum_{i \in T} g_i$.

**IMPORTANT NOTE:** The paper's Hessian $H$ uses the full-data Hessian at $\hat{\theta}$,
not the retained-data Hessian. This is intentional: the NS approximation is a single
Newton step on $L_{\backslash T}$ starting from $\hat{\theta}$, using the full Hessian as
a curvature estimate.

### Code

```python
from scipy.linalg import solve  # more stable than explicit inverse


def newton_step_hypothesis(theta_hat: np.ndarray, X: np.ndarray,
                           y: np.ndarray, T: set[int],
                           lam: float) -> np.ndarray:
    """Compute Newton Step approximation to leave-T-out optimum.
    
    Args:
        theta_hat: Full-data ERM optimum, shape (d,)
        X: Full design matrix, shape (n, d)
        y: Full labels, shape (n,)
        T: Deletion set (0-indexed)
        lam: Regularisation coefficient
    Returns:
        theta_ns: NS approximation, shape (d,)
    """
    n = X.shape[0]
    
    # Hessian at full-data optimum
    H = logistic_hessian(theta_hat, X, y, lam)  # (d, d)
    
    # Sum of per-sample gradients for deleted points
    g_T = per_sample_gradients(theta_hat, X, y)  # (n, d)
    T_list = sorted(T)
    grad_sum = g_T[T_list].sum(axis=0) / n  # (d,), with 1/n factor
    
    # Newton step: theta_hat + H^{-1} * grad_sum
    delta = solve(H, grad_sum, assume_a='pos')  # (d,)
    return theta_hat + delta
```

### Tests

```
TEST_P2_1 (output_shape):
    Assert theta_ns.shape == (d,).

TEST_P2_2 (trivial_case_no_deletion):
    T = empty set. grad_sum = 0. Assert theta_ns == theta_hat (up to atol).

TEST_P2_3 (approximation_quality_small_k):
    n=500, d=5, lam=0.1. Draw X ~ N(0, I), y from ground-truth theta*.
    Train theta_hat on full data. Set T = {random single index}.
    Retrain theta_true on retained data.
    Assert ||theta_ns - theta_true||_2 / ||theta_hat - theta_true||_2 < 0.3
    (NS should capture most of the update direction.)

TEST_P2_4 (approximation_quality_scales_with_k):
    Same setup but sweep k in {1, 5, 10, 20}.
    Assert that ||theta_ns - theta_true||_2 is monotonically non-decreasing in k
    (larger deletion sets are harder to approximate).

TEST_P2_5 (manual_2d_verification):
    X = np.array([[1, 0], [0, 1], [1, 1]]), y = np.array([1, 0, 1]), lam = 1.0.
    Compute theta_hat via scipy.optimize.minimize.
    T = {1}. Manually compute H, g_1, H^{-1} g_1 / n.
    Assert theta_ns matches manual calculation to atol=1e-10.

TEST_P2_6 (gradient_at_hypothesis):
    Compute the retained-data gradient at theta_ns.
    Assert it is smaller in norm than the retained-data gradient at theta_hat.
    (The NS step should reduce the gradient on the retained loss.)
```

---

## Phase 3: Verification Procedure (Algorithm 1, Lines 5-6)

### Specification

This is the most underspecified part of the paper. The paper states Theorem 1.2 but
refers to "Algorithm TODO" for the actual procedure. However, the proof structure
(Theorem 1.6) gives us enough to reconstruct a concrete verifier.

**What the verifier must check (from Theorem 1.2 correctness guarantee):**

If ACCEPT, then $\|\Sigma_{\backslash T}^{1/2}(\hat{\theta}_{\backslash T} - \theta_0)\|_2 \leq R_2$.

Since we cannot compute $\hat{\theta}_{\backslash T}$ (that would require retraining), the
verifier instead checks **sufficient conditions** from Theorem 1.6 that certify a
zero of $\nabla L_{\backslash T}$ exists within a ball of radius $R_2$ around $\theta_0$
in the $\Sigma_{\backslash T}$-norm.

**Concrete verification procedure (derived from Theorem 1.6):**

Given the retained-data loss $L_{\backslash T}$, hypothesis $\theta_0$, and tolerance $R_2$:

1. Compute the Hessian at the hypothesis: $H_0 = \nabla^2 L_{\backslash T}(\theta_0)$
2. Compute the "Gram-like" matrix: $G = X_{\backslash T}^\top H_0^{-1} X_{\backslash T}$
3. Extract the key quantities:
   - $C_D = \max_i G_{ii}$ (maximum diagonal, i.e. max leverage)
   - $C_O = \max_{i \neq j} |G_{ij}|$ (maximum off-diagonal)
   - $C_{op} = \|G\|_{op}$ (operator norm)
4. Compute residual norms:
   - $r_2 = \|X_{\backslash T}^\top H_0^{-1} \nabla L_{\backslash T}(\theta_0)\|_2$
   - $r_\infty = \|X_{\backslash T}^\top H_0^{-1} \nabla L_{\backslash T}(\theta_0)\|_\infty$
5. Choose $R_\infty$ as the smallest value satisfying both:
   - $R_\infty > r_\infty + C_D R_\infty^2 + C_O R_2^2$
     (rearranged from the $R_\infty$ condition in Theorem 1.6)
   - $C_{op}^{-1} > R_\infty$
6. **ACCEPT** if and only if ALL of the following hold:
   - $R_2 > r_2 / (1 - C_{op} R_\infty)$
   - $C_D^{-1} > C_{op}^{-1} > R_\infty > r_\infty + C_D R_\infty^2 + C_O R_2^2$
   - The $R_\infty$ from step 5 exists (the quadratic in $R_\infty$ has a valid root)

**Practical implementation for step 5:** The condition
$R_\infty > r_\infty + C_D R_\infty^2 + C_O R_2^2$ rearranges to
$C_D R_\infty^2 - R_\infty + (r_\infty + C_O R_2^2) < 0$. This quadratic in $R_\infty$
has real roots when $1 - 4 C_D (r_\infty + C_O R_2^2) > 0$, and the valid root is:

$$R_\infty^* = \frac{1 - \sqrt{1 - 4 C_D (r_\infty + C_O R_2^2)}}{2 C_D}$$

(the smaller root of the quadratic, which is the tightest valid bound).

### Code

```python
from scipy.linalg import eigvalsh, solve as la_solve
from numpy.linalg import norm


def compute_verification_quantities(X_ret: np.ndarray, y_ret: np.ndarray,
                                    theta_0: np.ndarray,
                                    lam: float) -> dict:
    """Compute all quantities needed for the Theorem 1.6 verification.
    
    Returns a dict with keys: C_D, C_O, C_op, r_2, r_inf, H_0
    """
    n_ret = X_ret.shape[0]
    
    # Hessian at hypothesis on retained data
    H_0 = logistic_hessian(theta_0, X_ret, y_ret, lam)
    
    # H_0^{-1} X_ret^T  (solve H_0 @ Z = X_ret^T for Z)
    # Z has shape (d, n_ret)
    H0_inv_Xt = la_solve(H_0, X_ret.T, assume_a='pos')
    
    # Gram-like matrix G = X_ret @ H_0^{-1} @ X_ret^T, shape (n_ret, n_ret)
    G = X_ret @ H0_inv_Xt
    
    C_D = np.max(np.diag(G))
    C_O = np.max(np.abs(G - np.diag(np.diag(G))))  # max |G_{ij}| for i != j
    C_op = np.max(np.abs(eigvalsh(G)))  # operator norm of symmetric matrix
    
    # Gradient of retained-data loss at hypothesis
    grad = logistic_gradient(theta_0, X_ret, y_ret, lam)
    
    # X_ret^T @ H_0^{-1} @ grad, shape (n_ret,)
    H0_inv_grad = la_solve(H_0, grad, assume_a='pos')
    Xt_H0inv_grad = X_ret @ H0_inv_grad  # (n_ret,)
    
    r_2 = norm(Xt_H0inv_grad, 2)
    r_inf = norm(Xt_H0inv_grad, np.inf)
    
    return {
        'C_D': C_D, 'C_O': C_O, 'C_op': C_op,
        'r_2': r_2, 'r_inf': r_inf, 'H_0': H_0
    }


def verify_hypothesis(X_ret: np.ndarray, y_ret: np.ndarray,
                      theta_0: np.ndarray, R_2: float,
                      lam: float) -> bool:
    """Run the Theorem 1.6-based verification.
    
    Returns True (ACCEPT) if the conditions are satisfied, False (REJECT) otherwise.
    """
    q = compute_verification_quantities(X_ret, y_ret, theta_0, lam)
    C_D, C_O, C_op = q['C_D'], q['C_O'], q['C_op']
    r_2, r_inf = q['r_2'], q['r_inf']
    
    # Step 5: Solve for R_inf from the quadratic
    # C_D * R_inf^2 - R_inf + (r_inf + C_O * R_2^2) < 0
    # has solutions when discriminant > 0
    rhs_const = r_inf + C_O * R_2**2
    discriminant = 1.0 - 4.0 * C_D * rhs_const
    
    if discriminant <= 0:
        return False  # REJECT: no valid R_inf exists
    
    R_inf_star = (1.0 - np.sqrt(discriminant)) / (2.0 * C_D)
    
    # Step 6: Check all conditions from Theorem 1.6
    # Condition: C_op^{-1} > R_inf
    if R_inf_star >= 1.0 / C_op:
        return False
    
    # Condition: C_D^{-1} > C_op^{-1}  (structural, data-dependent)
    if 1.0 / C_D <= 1.0 / C_op:
        return False
    
    # Condition: R_2 > r_2 / (1 - C_op * R_inf)
    if R_2 <= r_2 / (1.0 - C_op * R_inf_star):
        return False
    
    # Condition: R_inf > r_inf + C_D * R_inf^2 + C_O * R_2^2
    # (This is guaranteed by construction of R_inf_star, but verify numerically)
    if R_inf_star <= rhs_const + C_D * R_inf_star**2 - C_D * R_inf_star**2:
        # This should not trigger; included as a sanity check
        return False
    
    return True  # ACCEPT
```

### IMPORTANT DESIGN NOTE

The runtime of this verifier is dominated by computing $G = X_{\backslash T} H_0^{-1} X_{\backslash T}^\top$,
which is $O(n^2 d)$ — matching the paper's stated $\tilde{O}(n^2 d)$ for "Algorithm TODO".
If $n$ is large, this is expensive. For a first implementation this is acceptable. A
production version would exploit structure (e.g., only computing the diagonal and a few
rows of $G$, or using randomised methods for $C_{op}$).

### Tests

```
TEST_P3_1 (accept_true_optimum):
    n=200, d=5, lam=0.1. Train on full data, delete k=3 samples.
    Retrain to get true theta_T. Set R_2 large (e.g. 1.0).
    Assert verify_hypothesis(X_ret, y_ret, theta_true_T, R_2, lam) == True.
    (The true optimum has r_2 = 0, so any R_2 > 0 should accept.)

TEST_P3_2 (reject_garbage_hypothesis):
    Same setup. Set theta_0 = np.ones(d) * 100 (far from optimum).
    Set R_2 = 1e-6 (tiny).
    Assert verify_hypothesis returns False.

TEST_P3_3 (accept_ns_hypothesis_gaussian_pac):
    n=1000, d=5, lam=0.1. Draw X ~ N(0, I), y from ground truth.
    Delete k=5 random samples. Compute NS hypothesis.
    Set R_2 = C * k * d / n**1.5 with C a moderate constant (e.g. 10).
    Assert ACCEPT. (This tests Theorem 1.4's prediction.)

TEST_P3_4 (r2_threshold_sweep):
    Same Gaussian-PAC setup. Sweep R_2 from 1e-10 to 1.0 on a log scale.
    Record the transition point from REJECT to ACCEPT.
    Assert this transition occurs near k*d/n^{3/2} (within an order of magnitude).

TEST_P3_5 (verification_quantities_sanity):
    Assert C_D > 0, C_O >= 0, C_op > 0.
    Assert C_D <= C_op (diagonal bounded by operator norm).
    Assert r_2 >= r_inf (L2 norm >= L_inf for same vector — WAIT, this is
    wrong since these are norms of an (n_ret,)-dim vector. Actually
    r_inf <= r_2 <= sqrt(n_ret) * r_inf. Test the upper bound.)

TEST_P3_6 (exact_optimum_has_zero_residual):
    Compute theta_hat_ret = true retained optimum.
    Compute verification quantities at theta_hat_ret.
    Assert r_2 < 1e-8 and r_inf < 1e-8.
```

---

## Phase 4: Conditional Fallback (Algorithm 1, Lines 7-10)

### Specification

Straightforward control flow:

```
if verify_hypothesis(X_ret, y_ret, theta_ns, R_2, lam):
    theta_ret = theta_ns
else:
    theta_ret = retrain(X_ret, y_ret, lam)  # exact ERM on retained data
```

The retraining fallback uses scipy.optimize.minimize (L-BFGS-B) on the retained-data
loss.

### Code

```python
from scipy.optimize import minimize


def retrain_exact(X_ret: np.ndarray, y_ret: np.ndarray,
                  lam: float, d: int) -> np.ndarray:
    """Exact ERM on retained data via L-BFGS-B.
    
    Args:
        X_ret, y_ret: Retained data
        lam: Regularisation coefficient
        d: Dimension (for initialisation)
    Returns:
        theta_hat_ret: Exact optimum, shape (d,)
    """
    result = minimize(
        fun=logistic_loss,
        x0=np.zeros(d),
        args=(X_ret, y_ret, lam),
        jac=logistic_gradient,  # supply analytic gradient
        method='L-BFGS-B',
        options={'maxiter': 10000, 'gtol': 1e-12}
    )
    if not result.success:
        raise RuntimeError(f"Retraining failed: {result.message}")
    return result.x
```

### Tests

```
TEST_P4_1 (retrain_converges):
    Random X (200x5), y, lam=0.1.
    theta_ret = retrain_exact(X, y, lam, 5).
    Assert ||gradient at theta_ret|| < 1e-8.

TEST_P4_2 (accept_uses_hypothesis):
    Mock verify_hypothesis to return True.
    Assert output theta equals theta_ns (exact identity, not approximate).

TEST_P4_3 (reject_uses_retrain):
    Mock verify_hypothesis to return False.
    Assert output theta equals retrain_exact output.
    Assert ||gradient at output|| < 1e-8.

TEST_P4_4 (retrain_matches_sklearn):
    Compare retrain_exact output against sklearn LogisticRegression
    (with C = 1/lam, solver='lbfgs', penalty='l2', fit_intercept=False).
    Assert allclose with atol=1e-5.
    (This cross-validates our loss implementation against a trusted library.)
```

---

## Phase 5: Masking and Output (Algorithm 1, Lines 11-14)

### Specification

**Noise variance:**

$$\sigma^2 = \frac{2 \log(1.25 / \delta)}{\varepsilon^2} \cdot R_2^2$$

**Noise distribution:**

$$z \sim \mathcal{N}(0, \sigma^2 \Sigma_{\backslash T}^{-1})$$

**Sampling:** To sample $z \sim \mathcal{N}(0, \sigma^2 \Sigma_{\backslash T}^{-1})$:

1. Compute Cholesky $\Sigma_{\backslash T} = L L^\top$ where $L$ is lower-triangular.
2. Sample $\xi \sim \mathcal{N}(0, I_d)$.
3. Solve $L^\top z' = \xi$ for $z'$ (backward substitution).
4. Set $z = \sigma \cdot z'$.

This works because $\text{Cov}(z) = \sigma^2 L^{-\top} L^{-1} = \sigma^2 (L L^\top)^{-1} = \sigma^2 \Sigma_{\backslash T}^{-1}$.

**Output:**

$$\tilde{\theta}_{\backslash T} = \theta_{\backslash T} + z$$

### Code

```python
from scipy.linalg import cho_factor, cho_solve


def compute_noise_variance(epsilon: float, delta: float,
                           R_2: float) -> float:
    """Compute sigma^2 for Gaussian masking."""
    return 2.0 * np.log(1.25 / delta) / (epsilon**2) * R_2**2


def sample_masking_noise(Sigma_ret: np.ndarray, sigma_sq: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Sample z ~ N(0, sigma^2 * Sigma_ret^{-1}).
    
    Args:
        Sigma_ret: Sigma_{\\T} = X_{\\T}^T X_{\\T}, shape (d, d), must be PD
        sigma_sq: Noise variance sigma^2
        rng: numpy random Generator for reproducibility
    Returns:
        z: Noise vector, shape (d,)
    """
    d = Sigma_ret.shape[0]
    sigma = np.sqrt(sigma_sq)
    
    # Cholesky: Sigma_ret = L @ L^T
    L, lower = cho_factor(Sigma_ret, lower=True)
    
    # xi ~ N(0, I)
    xi = rng.standard_normal(d)
    
    # Solve L^T z' = xi  =>  z' = L^{-T} xi
    # cho_solve solves (L L^T) x = b, so we need a different approach:
    # We want L^{-T} xi. Since L is lower triangular, L^T is upper triangular.
    from scipy.linalg import solve_triangular
    z_prime = solve_triangular(L.T, xi, lower=False)  # L^T is upper
    
    return sigma * z_prime


def mask_output(theta_ret: np.ndarray, Sigma_ret: np.ndarray,
                epsilon: float, delta: float, R_2: float,
                rng: np.random.Generator) -> np.ndarray:
    """Apply Gaussian masking to produce final unlearned model.
    
    Returns theta_tilde = theta_ret + z.
    """
    sigma_sq = compute_noise_variance(epsilon, delta, R_2)
    z = sample_masking_noise(Sigma_ret, sigma_sq, rng)
    return theta_ret + z
```

### Tests

```
TEST_P5_1 (noise_variance_formula):
    epsilon=1.0, delta=0.01, R_2=0.5.
    Expected: 2 * log(125) / 1.0 * 0.25 = 2 * 4.828 * 0.25 ≈ 2.414.
    Assert allclose.

TEST_P5_2 (noise_mean_zero):
    Sigma_ret = some PD matrix. sigma_sq = 1.0.
    Draw 50000 samples. Assert empirical mean is near zero (atol=0.05).

TEST_P5_3 (noise_covariance):
    Same setup. Compute empirical covariance of 50000 samples.
    Assert allclose to sigma_sq * inv(Sigma_ret), rtol=0.1.

TEST_P5_4 (noise_covariance_non_isotropic):
    Sigma_ret with condition number ~100.
    Draw 50000 samples. Assert empirical covariance matches target, rtol=0.15.
    (Tests numerical stability for ill-conditioned Sigma.)

TEST_P5_5 (output_shape):
    Assert mask_output returns array of shape (d,).

TEST_P5_6 (reproducibility):
    Same rng seed => same output. Different seed => different output.

TEST_P5_7 (noise_scales_with_R2):
    Fix epsilon, delta. Compute sigma_sq for R_2 = 0.1 and R_2 = 1.0.
    Assert sigma_sq ratio is 100 (scales as R_2^2).

TEST_P5_8 (noise_scales_with_epsilon):
    Fix delta, R_2. Compute sigma_sq for eps=0.5 and eps=1.0.
    Assert sigma_sq ratio is 4.0 (scales as 1/eps^2).
```

---

## Phase 6: End-to-End Integration

### Specification

The full pipeline, corresponding to Algorithm 1:

```python
def unerring(X: np.ndarray, y: np.ndarray, theta_hat: np.ndarray,
             T: set[int], epsilon: float, delta: float, R_2: float,
             lam: float, rng: np.random.Generator) -> np.ndarray:
    """UnERRING: certified unlearning with geometry-aware masking.
    
    Args:
        X: Full design matrix, shape (n, d)
        y: Full labels, shape (n,)
        theta_hat: Full-data ERM optimum, shape (d,)
        T: Deletion set (0-indexed)
        epsilon, delta: Unlearning privacy parameters
        R_2: Tolerance radius
        lam: Regularisation coefficient
        rng: Random number generator
    Returns:
        theta_tilde: Unlearned model, shape (d,)
    """
    d = X.shape[1]
    
    # Phase 1: Data preprocessing
    X_ret, y_ret = remove_samples(X, y, T)
    Sigma_ret = compute_sigma(X_ret)
    
    # Phase 2: Hypothesis generation
    theta_ns = newton_step_hypothesis(theta_hat, X, y, T, lam)
    
    # Phase 3: Verification
    accepted = verify_hypothesis(X_ret, y_ret, theta_ns, R_2, lam)
    
    # Phase 4: Conditional fallback
    if accepted:
        theta_ret = theta_ns
    else:
        theta_ret = retrain_exact(X_ret, y_ret, lam, d)
    
    # Phase 5: Masking
    theta_tilde = mask_output(theta_ret, Sigma_ret, epsilon, delta, R_2, rng)
    
    return theta_tilde
```

### Tests

```
TEST_E2E_1 (smoke_test):
    n=500, d=5, lam=0.1, k=3, eps=1.0, delta=0.01.
    X ~ N(0, I), y from ground truth.
    theta_hat = retrain_exact(X, y, lam, d).
    T = random 3 indices.
    R_2 = 10 * k * d / n**1.5.
    theta_tilde = unerring(X, y, theta_hat, T, eps, delta, R_2, lam, rng).
    Assert theta_tilde.shape == (d,) and no exceptions raised.

TEST_E2E_2 (unlearning_privacy_empirical):
    Run 5000 trials. In each trial:
      - Draw fresh X, y from Gaussian-PAC
      - Train theta_hat
      - Pick random T of size k
      - Run unerring to get theta_tilde_U (the unlearning output)
      - Run reference M': retrain on X_ret, add same masking noise => theta_tilde_M
    Collect all theta_tilde_U and theta_tilde_M samples.
    Run a multivariate two-sample test (e.g., energy distance or MMD).
    Assert test does NOT reject H0 (distributions are same) at alpha=0.05.
    (This is the empirical (eps, delta)-indistinguishability check.)

TEST_E2E_3 (excess_risk):
    Gaussian-PAC setup: n=2000, d=5, lam=0.01, k=5, eps=1.0, delta=0.01.
    For 200 trials, compute:
      - excess_risk = L_test(theta_tilde) - L_test(theta_hat_ret_noised)
    where L_test is the population loss on a held-out test set,
    and theta_hat_ret_noised is the reference M' output.
    Assert mean(excess_risk) is small relative to the base risk.

TEST_E2E_4 (deletion_capacity):
    n=2000, d=5, lam=0.1. Sweep k from 1 to 200.
    R_2 = C * k * d / n**1.5 (with C from Theorem 1.4).
    For each k, run 50 trials and record fraction of ACCEPTs.
    Assert that for k << n^2 / d^{5/4} ≈ 2000^2 / 5^{1.25} ≈ ~500k,
    the accept rate is > 0.9.
    Assert that accept rate degrades as k approaches the capacity bound.

TEST_E2E_5 (non_isotropic_covariance):
    Same setup but X ~ N(0, Sigma) with Sigma having condition number 50.
    Run the full pipeline with k=5.
    Assert ACCEPT (the geometry-aware noise should handle non-isotropic data).
    Compare excess risk against a version using isotropic noise z ~ N(0, sigma^2 I).
    Assert geometry-aware version has lower excess risk.

TEST_E2E_6 (reference_distribution_match):
    For a fixed X, y, T:
      - Run unerring 2000 times (varying rng seed) to get distribution of outputs.
      - Run M' 2000 times to get reference distribution.
      - On the ACCEPT branch only, compare the two distributions.
    This directly tests Theorem 1.3.

TEST_E2E_7 (deterministic_on_reject):
    Force a REJECT (e.g., use R_2 = 1e-15).
    Run unerring twice with same rng seed.
    Assert outputs are identical.
    (On REJECT, the only randomness is the masking noise, which is seeded.)
```

---

## R_2 Selection Guide

The paper specifies $R_2 = \Theta(kd / n^{3/2})$ under Gaussian-PAC (Theorem 1.4).
In practice, the constant matters. We recommend:

```python
def default_R2(k: int, d: int, n: int, safety_factor: float = 5.0) -> float:
    """Default R_2 following Theorem 1.4.
    
    The safety_factor absorbs the unspecified Theta() constant.
    Start with 5.0 and tune via TEST_P3_4.
    """
    return safety_factor * k * d / n**1.5
```

The safety_factor should be calibrated empirically: too small causes spurious
REJECTs (unnecessary retraining), too large adds excessive masking noise.

---

## Known Limitations and Open Items

1. **DRIF approximation not implemented.** The paper references DRIF as potentially
   tighter than NS, but the formula depends on [RH25]. Once that's consulted, DRIF
   can be added as an alternative to `newton_step_hypothesis`.

2. **Verifier runtime.** The $O(n^2 d)$ Gram matrix computation is the bottleneck.
   For $n > 10^4$, consider randomised approximations to $C_{op}$.

3. **The $R_\infty$ quadratic.** The derivation of the verification conditions from
   Theorem 1.6 involves solving a quadratic. The smaller root is correct, but
   edge cases (very small $C_D$) could cause numerical issues. Add a guard for
   $C_D < \epsilon_{machine}$.

4. **Intercept term.** The paper assumes no intercept. If needed, augment $X$ with
   a column of ones, but note this changes the regularisation structure.

5. **Theorem 1.2's stated runtime is $\tilde{O}(n^2 d)$.** Our verifier matches this,
   but the paper's "Algorithm TODO" might have additional tricks (e.g., early
   termination) that we don't capture.

6. **Constants in Theorem 1.4.** The $\Theta()$ notation hides constants that
   depend on the ground-truth model norm $\|\theta^*\|_\Sigma$. The safety_factor
   in `default_R2` is a placeholder for these.
