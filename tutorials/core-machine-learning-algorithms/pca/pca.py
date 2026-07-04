import numpy as np


class PCA:
    """Principal Component Analysis via covariance eigendecomposition."""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # Center the data around the feature means.
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # Covariance matrix (features x features).
        cov = np.cov(Xc, rowvar=False)

        # Eigendecompose the symmetric covariance matrix.
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenpairs by eigenvalue, descending.
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Keep the top n_components principal directions.
        self.components_ = eigvecs[:, : self.n_components]
        self.explained_variance_ = eigvals[: self.n_components]
        self.explained_variance_ratio_ = eigvals[: self.n_components] / eigvals.sum()
        return self

    def transform(self, X):
        # Project centered data onto the principal axes.
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        # Map projections back to the original space.
        return Z @ self.components_.T + self.mean_


if __name__ == "__main__":
    np.random.seed(0)

    # Synthetic 3D data that mostly lives on a 2D plane.
    n = 200
    t = np.random.randn(n, 2)
    W = np.array([[2.0, 0.0], [0.5, 1.5], [1.0, -1.0]])  # 3x2 mixing
    X = t @ W.T + 0.05 * np.random.randn(n, 3)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    X_rec = pca.inverse_transform(Z)

    err = np.mean((X - X_rec) ** 2)
    print("Original shape:", X.shape, "-> reduced shape:", Z.shape)
    print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
    print("Total variance kept: {:.4f}".format(pca.explained_variance_ratio_.sum()))
    print("Reconstruction MSE (3D->2D->3D): {:.6f}".format(err))
