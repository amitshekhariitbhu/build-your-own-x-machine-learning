import numpy as np


class LDA:
    """Linear Discriminant Analysis for supervised dimensionality reduction."""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        n_features = X.shape[1]
        classes = np.unique(y)
        mean_all = X.mean(axis=0)

        # Within-class scatter S_W and between-class scatter S_B.
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        for c in classes:
            Xc = X[y == c]
            mean_c = Xc.mean(axis=0)

            # Scatter within class c (sum of outer products of centered points).
            diff = Xc - mean_c
            S_W += diff.T @ diff

            # Scatter between class mean and global mean, weighted by class size.
            md = (mean_c - mean_all).reshape(-1, 1)
            S_B += len(Xc) * (md @ md.T)

        # Solve the generalized problem via eigendecomposition of inv(S_W) @ S_B.
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
        eigvals, eigvecs = eigvals.real, eigvecs.real

        # Keep the top n_components discriminant directions.
        order = np.argsort(eigvals)[::-1]
        self.components_ = eigvecs[:, order[: self.n_components]]
        return self

    def transform(self, X):
        # Project data onto the discriminant axes.
        return X @ self.components_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


if __name__ == "__main__":
    np.random.seed(0)

    # Synthetic 3-class data in 4D with distinct class means.
    n_per = 60
    means = np.array([[0, 0, 0, 0], [4, 4, 1, 0], [8, 0, 2, 1]], dtype=float)
    X = np.vstack([m + np.random.randn(n_per, 4) for m in means])
    y = np.repeat([0, 1, 2], n_per)

    # Shuffle then split into train/test.
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(0.7 * len(X))
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

    # Reduce 4D -> 2D using the training set.
    lda = LDA(n_components=2)
    Ztr = lda.fit_transform(Xtr, ytr)
    Zte = lda.transform(Xte)

    # Classify test points by nearest projected class mean.
    classes = np.unique(ytr)
    proj_means = np.array([Ztr[ytr == c].mean(axis=0) for c in classes])
    dists = np.linalg.norm(Zte[:, None, :] - proj_means[None, :, :], axis=2)
    pred = classes[np.argmin(dists, axis=1)]

    acc = np.mean(pred == yte)
    print("Original shape:", Xtr.shape, "-> reduced shape:", Ztr.shape)
    print("Test samples:", len(yte))
    print("Nearest-projected-mean accuracy: {:.4f}".format(acc))
