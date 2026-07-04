import numpy as np


class MovieRecommender:
    """Matrix factorization via SGD: R ~ global + b_u + b_i + P @ Q.T."""

    def __init__(self, n_factors=8, lr=0.01, reg=0.05, n_epochs=40, seed=0):
        self.n_factors = n_factors
        self.lr = lr            # SGD step size
        self.reg = reg          # L2 regularization strength
        self.n_epochs = n_epochs
        self.seed = seed

    def fit(self, R, mask):
        # R: user x movie ratings, mask: 1 where a rating is observed.
        rng = np.random.RandomState(self.seed)
        n_users, n_items = R.shape
        self.R = R
        self.mask = mask

        # Learnable parameters: biases and latent factor matrices.
        self.mu = R[mask == 1].mean()               # global mean rating
        self.bu = np.zeros(n_users)                 # per-user bias
        self.bi = np.zeros(n_items)                 # per-movie bias
        self.P = 0.1 * rng.randn(n_users, self.n_factors)   # user factors
        self.Q = 0.1 * rng.randn(n_items, self.n_factors)   # movie factors

        # Observed (user, item) index pairs to iterate over.
        us, is_ = np.where(mask == 1)

        for _ in range(self.n_epochs):
            order = rng.permutation(len(us))        # shuffle each epoch
            for idx in order:
                u, i = us[idx], is_[idx]
                pred = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                err = R[u, i] - pred

                # Gradient step on the regularized squared error.
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * pu - self.reg * self.Q[i])
        return self

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]

    def predict_all(self):
        # Full reconstructed rating matrix.
        return self.mu + self.bu[:, None] + self.bi[None, :] + self.P @ self.Q.T

    def recommend(self, u, n=5):
        # Top-n movies the user has NOT rated in the training set.
        scores = self.predict_all()[u]
        scores[self.mask[u] == 1] = -np.inf          # hide already-seen movies
        return np.argsort(scores)[::-1][:n]


if __name__ == "__main__":
    np.random.seed(0)

    # Planted low-rank structure: true user/movie factors + biases + noise.
    n_users, n_items, true_k = 100, 60, 4
    U = np.random.randn(n_users, true_k)
    V = np.random.randn(n_items, true_k)
    bu_true = 0.5 * np.random.randn(n_users)
    bi_true = 0.5 * np.random.randn(n_items)
    mu_true = 3.5
    ground = mu_true + bu_true[:, None] + bi_true[None, :] + U @ V.T
    R_full = ground + 0.3 * np.random.randn(n_users, n_items)   # observed ratings
    R_full = np.clip(R_full, 1.0, 5.0)

    # Hold out 20% of entries as a test split.
    test_mask = (np.random.rand(n_users, n_items) < 0.2).astype(int)
    train_mask = 1 - test_mask

    model = MovieRecommender(n_factors=8, lr=0.02, reg=0.05, n_epochs=40)
    model.fit(R_full * train_mask, train_mask)

    # Held-out RMSE vs. the global-mean baseline.
    pred = model.predict_all()
    ti, tj = np.where(test_mask == 1)
    truth = R_full[ti, tj]
    mf_rmse = np.sqrt(np.mean((truth - pred[ti, tj]) ** 2))
    base_rmse = np.sqrt(np.mean((truth - R_full[train_mask == 1].mean()) ** 2))

    # Correlation between predictions and the noise-free planted ground truth.
    corr = np.corrcoef(pred[ti, tj], ground[ti, tj])[0, 1]

    print("Users x Movies:      %d x %d  (true rank %d)" % (n_users, n_items, true_k))
    print("Held-out RMSE (MF):  %.4f" % mf_rmse)
    print("Baseline RMSE (mean):%.4f" % base_rmse)
    print("Improvement:         %.1f%%" % (100 * (1 - mf_rmse / base_rmse)))
    print("Corr(pred, planted): %.4f" % corr)
    print("Top-5 for user 0:    %s" % model.recommend(0, n=5).tolist())
    print("MF beats baseline:   %s" % (mf_rmse < base_rmse))
