"""NNDSVD initialization of the non-negative imaging decoder.

Adapted from the opNMF implementation in pyOPNMF (see README for credit).

Reference:
    C. Boutsidis and E. Gallopoulos, "SVD-based initialization: A head start
    for nonnegative matrix factorization", Pattern Recognition 41 (2008).
"""
import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import svds


def nndsvd(X, n_components):
    """Non-negative double SVD initialization.

    Args:
        X: non-negative (n_voxels, n_subjects) data matrix.
        n_components: number of components.

    Returns:
        W: (n_voxels, n_components) non-negative matrix
        H: (n_components, n_subjects) non-negative matrix
    """
    if not np.isfinite(X).all():
        raise ValueError("Input matrix contains NaN or Inf values.")
    if (X < 0).any():
        raise ValueError("Input matrix must be non-negative.")

    n_features, n_samples = X.shape
    W = np.zeros((n_features, n_components), dtype=X.dtype)
    H = np.zeros((n_components, n_samples), dtype=X.dtype)

    # Truncated SVD; scipy returns singular values in ascending order.
    U, S, Vt = svds(X, k=n_components)
    order = np.argsort(S)[::-1]
    U, S, V = U[:, order], S[order], Vt[order].T

    # First singular triplet can be chosen non-negative.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[:, 0])

    for i in range(1, n_components):
        u, v = U[:, i], V[:, i]
        up, un = np.clip(u, 0, None), np.abs(np.clip(u, None, 0))
        vp, vn = np.clip(v, 0, None), np.abs(np.clip(v, None, 0))
        n_up, n_vp, n_un, n_vn = norm(up), norm(vp), norm(un), norm(vn)
        term_p, term_n = n_up * n_vp, n_un * n_vn
        if term_p >= term_n:
            W[:, i] = np.sqrt(S[i] * term_p) * up / n_up
            H[i, :] = np.sqrt(S[i] * term_p) * vp / n_vp
        else:
            W[:, i] = np.sqrt(S[i] * term_n) * un / n_un
            H[i, :] = np.sqrt(S[i] * term_n) * vn / n_vn

    # Replace (near-)zero entries, as in the original implementation.
    W[W < 1e-10] = 0.1
    H[H < 1e-10] = 0.1
    return W, H
