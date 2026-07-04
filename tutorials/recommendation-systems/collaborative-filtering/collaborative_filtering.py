import numpy as np


class CollaborativeFiltering:
    """Matrix-completion CF: fill a partially observed rating matrix by
    iterated truncated-SVD reconstruction (mean-centered low-rank fit)."""

    def __init__(self, rank=4, n_iters=30, seed=0):
        self.rank = rank          # target latent dimension (k)
        self.n_iters = n_iters    # imputation refinement passes
        self.seed = seed

    def fit(self, R, mask):
        # R: user x item ratings, mask: 1 where a rating is observed.
        self.mask = mask
        obs = mask == 1

        # Global mean of observed entries; the low-rank part models residuals.
        self.mu = R[obs].mean()

        # Start with observed values in place and the mean elsewhere.
        X = np.where(obs, R, self.mu)

        for _ in range(self.n_iters):
            # Low-rank reconstruction of the current fully-filled matrix.
            U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
            k = self.rank
            low_rank = (U[:, :k] * S[:k]) @ Vt[:k]      # rank-k truncation
            X_hat = self.mu + low_rank

            # Keep known ratings exact; only overwrite the missing entries.
            X = np.where(obs, R, X_hat)

        # Cache final factors and the completed matrix.
        self.U, self.S, self.Vt = U[:, :k], S[:k], Vt[:k]
        self.full = X_hat
        return self

    def predict(self, u, i):
        return self.full[u, i]

    def predict_all(self):
        # Full reconstructed rating matrix (predictions for every cell).
        return self.full

    def recommend(self, u, n=5):
        # Top-n items the user has NOT rated in the training set.
        scores = self.full[u].copy()
        scores[self.mask[u] == 1] = -np.inf           # hide already-seen items
        return np.argsort(scores)[::-1][:n]


if __name__ == "__main__":
    np.random.seed(0)

    # Planted low-rank structure: true user/item factors + global mean + noise.
    n_users, n_items, true_k = 120, 80, 4
    U_true = np.random.randn(n_users, true_k)
    V_true = np.random.randn(n_items, true_k)
    mu_true = 3.5
    ground = mu_true + U_true @ V_true.T                # noise-free signal
    R_full = ground + 0.3 * np.random.randn(n_users, n_items)
    R_full = np.clip(R_full, 1.0, 5.0)

    # Hold out 20% of entries as a test split.
    test_mask = (np.random.rand(n_users, n_items) < 0.2).astype(int)
    train_mask = 1 - test_mask

    model = CollaborativeFiltering(rank=true_k, n_iters=30)
    model.fit(R_full * train_mask, train_mask)

    # Held-out RMSE vs. the global-mean baseline.
    pred = model.predict_all()
    ti, tj = np.where(test_mask == 1)
    truth = R_full[ti, tj]
    cf_rmse = np.sqrt(np.mean((truth - pred[ti, tj]) ** 2))
    base_rmse = np.sqrt(np.mean((truth - R_full[train_mask == 1].mean()) ** 2))

    # Correlation between predictions and the noise-free planted ground truth.
    corr = np.corrcoef(pred[ti, tj], ground[ti, tj])[0, 1]

    # Do the recovered latent factors span the planted subspace? Principal-angle
    # cosines between the two orthonormal bases (1.0 = identical subspace).
    Qb = np.linalg.qr(U_true)[0]                        # planted user subspace
    align = np.linalg.svd(model.U.T @ Qb, compute_uv=False).mean()
    rand_align = np.sqrt(true_k / n_users)             # ~expected for random subspaces

    print("Users x Items:        %d x %d  (true rank %d)" % (n_users, n_items, true_k))
    print("Held-out RMSE (CF):   %.4f" % cf_rmse)
    print("Baseline RMSE (mean): %.4f" % base_rmse)
    print("Improvement:          %.1f%%" % (100 * (1 - cf_rmse / base_rmse)))
    print("Corr(pred, planted):  %.4f" % corr)
    print("Subspace align:       %.3f  (random ~ %.3f)" % (align, rand_align))
    print("Top-5 for user 0:     %s" % model.recommend(0, n=5).tolist())
    print("CF beats baseline:    %s" % (cf_rmse < base_rmse))
