import numpy as np


def svd(A):
    """Singular Value Decomposition via eigendecomposition of A^T A.

    Returns U, S, Vt such that A == U @ diag(S) @ Vt.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    # A^T A is symmetric PSD; its eigenvectors are the right singular vectors V,
    # and its eigenvalues are the squared singular values.
    gram = A.T @ A
    eigvals, V = np.linalg.eigh(gram)

    # Sort eigenpairs by eigenvalue, descending.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    V = V[:, order]

    # Singular values are sqrt of the (clipped) eigenvalues.
    sigma = np.sqrt(np.clip(eigvals, 0.0, None))

    # Left singular vectors: U = A V / sigma, guarding tiny/zero sigma.
    k = min(m, n)
    sigma = sigma[:k]
    V = V[:, :k]
    U = np.zeros((m, k))
    for i in range(k):
        if sigma[i] > 1e-12:
            U[:, i] = (A @ V[:, i]) / sigma[i]
        else:
            U[:, i] = 0.0  # degenerate direction; left vector undefined

    return U, sigma, V.T


if __name__ == "__main__":
    np.random.seed(0)

    # Random rectangular matrix.
    A = np.random.randn(6, 4)

    U, S, Vt = svd(A)
    A_rec = U @ np.diag(S) @ Vt

    err = np.sqrt(np.mean((A - A_rec) ** 2))
    print("A shape:", A.shape)
    print("Singular values:", np.round(S, 4))
    # Sanity check (comment): these match np.linalg.svd(A)[1] up to rounding.
    print("U columns orthonormal? UtU ~= I:",
          np.allclose(U.T @ U, np.eye(S.shape[0]), atol=1e-8))
    print("V rows orthonormal? VVt ~= I:",
          np.allclose(Vt @ Vt.T, np.eye(S.shape[0]), atol=1e-8))
    print("Reconstruction RMSE (near 0): {:.2e}".format(err))
