"""Linear discriminant analysis with Ledoit-Wolf shrinkage, in numpy.

Shrinkage LDA is the standard classifier for small EEG data sets: one covariance
matrix shared by all classes, pulled towards a scaled identity by exactly as much
as its own estimation error warrants. The fit is closed-form, has nothing to
tune, and is fast enough to repeat a few thousand times for a shuffling test.

Having it here, rather than importing it, keeps training free of a dependency
whose saved objects stop loading when its version changes. The shrinkage
coefficient is tested against scikit-learn's.
"""

from __future__ import annotations

import numpy as np


def ledoit_wolf_shrinkage(centred: np.ndarray) -> float:
    """Optimal shrinkage of the covariance of zero-mean rows towards ``trace / p * I`` (Ledoit and Wolf, 2004)."""
    n, p = centred.shape
    if n < 2:
        return 1.0
    gram_trace = float((centred**2).sum())  # n * trace(S)
    mu = gram_trace / (n * p)
    if n >= p:
        s = centred.T @ centred / n
        s_norm2 = float((s**2).sum())
    else:  # the squared Frobenius norm of S equals that of the n x n Gram matrix, scaled
        gram = centred @ centred.T / n
        s_norm2 = float((gram**2).sum())
    delta = (s_norm2 - 2.0 * mu * gram_trace / n + p * mu**2) / p  # ||S - mu I||^2 / p
    row_norm4 = float(((centred**2).sum(axis=1) ** 2).sum())
    beta = (row_norm4 / n - s_norm2) / (n * p)
    beta = min(max(beta, 0.0), delta)
    return float(beta / delta) if delta > 0 else 0.0


def fit_lda(
    X: np.ndarray, y: np.ndarray, n_classes: int, shrinkage: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit on standardised features: ``(mean, scale, weights[K, D], bias[K])``.

    Scores ``((x - mean) / scale) @ weights.T + bias`` are log-posteriors up to a
    constant, with equal priors, so their softmax is the class probability. A class
    that does not occur in ``y`` gets a bias it can never win with.
    """
    present = np.unique(y)
    if len(present) < 2:
        raise ValueError("need at least two classes to fit a classifier")
    mean = X.mean(axis=0)
    scale = np.maximum(X.std(axis=0), 1e-9)
    Z = (X - mean) / scale
    n, p = Z.shape
    centres = np.zeros((n_classes, p))
    for k in present:
        centres[k] = Z[y == k].mean(axis=0)
    centred = Z - centres[y]
    gamma = ledoit_wolf_shrinkage(centred) if shrinkage is None else float(shrinkage)
    covariance = centred.T @ centred / n
    covariance = (1.0 - gamma) * covariance + gamma * (np.trace(covariance) / p) * np.eye(p)
    covariance.flat[:: p + 1] += 1e-10  # exactly singular only if a feature is constant and gamma is zero
    solved = np.linalg.solve(covariance, centres[present].T).T  # [present, p]
    weights = np.zeros((n_classes, p))
    bias = np.full(n_classes, -1e9)
    weights[present] = solved
    bias[present] = -0.5 * np.einsum("kp,kp->k", centres[present], solved)
    return mean, scale, weights, bias
