import numpy as np


class NetflixRecommender:
    """Netflix-Prize-style biased matrix factorization trained by SGD.

    Predicts a rating as:
        r_hat(u, i) = mu + b_u[u] + b_i[i] + dot(P[u], Q[i])
    where mu is the global mean, b_u / b_i are learned user / item biases,
    and P / Q are low-rank latent factor matrices. All parameters are fit
    from scratch with regularized stochastic gradient descent.
    """

    def __init__(self, n_factors=10, lr=0.01, reg=0.05, n_epochs=30, seed=0):
        self.n_factors = n_factors
        self.lr = lr            # SGD learning rate
        self.reg = reg          # L2 regularization strength
        self.n_epochs = n_epochs
        self.seed = seed

    def fit(self, R, mask):
        # R: users x items rating matrix; mask: 1 where a rating is observed.
        rng = np.random.RandomState(self.seed)
        n_users, n_items = R.shape
        self.mask = mask

        # Parameters: global mean, per-user/item biases, latent factors.
        self.mu = R[mask == 1].mean()
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.P = 0.1 * rng.randn(n_users, self.n_factors)
        self.Q = 0.1 * rng.randn(n_items, self.n_factors)

        us, is_ = np.where(mask == 1)           # observed (user, item) pairs
        for _ in range(self.n_epochs):
            for idx in rng.permutation(len(us)):
                u, i = us[idx], is_[idx]
                err = R[u, i] - self._predict(u, i)

                # Regularized gradient step on the squared error.
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * pu - self.reg * self.Q[i])
        return self

    def _predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]

    def predict(self, u, i):
        # Clip to the valid 1..5 star range like the Netflix Prize scale.
        return np.clip(self._predict(u, i), 1.0, 5.0)

    def predict_all(self):
        # Full reconstructed rating matrix (mu + biases + P Q^T).
        return self.mu + self.bu[:, None] + self.bi[None, :] + self.P @ self.Q.T

    def recommend(self, u, n=5):
        # Top-n items user u has NOT already rated in the training set.
        scores = self.predict_all()[u]
        scores[self.mask[u] == 1] = -np.inf
        return np.argsort(scores)[::-1][:n]


if __name__ == "__main__":
    np.random.seed(0)

    # --- Synthetic Netflix-like data with planted biases + latent factors ---
    n_users, n_items, true_k = 200, 100, 5
    U = np.random.randn(n_users, true_k)                 # true user factors
    V = np.random.randn(n_items, true_k)                 # true item factors
    bu_true = 0.8 * np.random.randn(n_users)             # generous/harsh raters
    bi_true = 0.8 * np.random.randn(n_items)             # popular/unpopular titles
    mu_true = 3.5
    ground = mu_true + bu_true[:, None] + bi_true[None, :] + U @ V.T
    R_full = np.clip(ground + 0.3 * np.random.randn(n_users, n_items), 1.0, 5.0)

    # Hold out 20% of the observed entries as a test split.
    test_mask = (np.random.rand(n_users, n_items) < 0.2).astype(int)
    train_mask = 1 - test_mask

    model = NetflixRecommender(n_factors=10, lr=0.02, reg=0.05, n_epochs=30)
    model.fit(R_full * train_mask, train_mask)

    # --- Held-out RMSE vs. two baselines ---
    ti, tj = np.where(test_mask == 1)
    truth = R_full[ti, tj]
    train_vals = R_full[train_mask == 1]

    pred_mf = np.clip(model.predict_all(), 1.0, 5.0)[ti, tj]
    mf_rmse = np.sqrt(np.mean((truth - pred_mf) ** 2))

    # Global-mean baseline: predict the same average rating everywhere.
    mu = train_vals.mean()
    mean_rmse = np.sqrt(np.mean((truth - mu) ** 2))

    # Bias-only baseline: mu + user bias + item bias, no latent factors.
    bu = np.array([R_full[u][train_mask[u] == 1].mean() - mu
                   if train_mask[u].sum() else 0.0 for u in range(n_users)])
    bi = np.array([(R_full[:, i] - mu)[train_mask[:, i] == 1].mean()
                   if train_mask[:, i].sum() else 0.0 for i in range(n_items)])
    pred_bias = np.clip(mu + bu[ti] + bi[tj], 1.0, 5.0)
    bias_rmse = np.sqrt(np.mean((truth - pred_bias) ** 2))

    print("Users x Items:          %d x %d  (true rank %d)" % (n_users, n_items, true_k))
    print("Baseline RMSE (mean):   %.4f" % mean_rmse)
    print("Baseline RMSE (biases): %.4f" % bias_rmse)
    print("Held-out RMSE (MF):     %.4f" % mf_rmse)
    print("Gain over mean:         %.1f%%" % (100 * (1 - mf_rmse / mean_rmse)))
    print("Gain over bias-only:    %.1f%%" % (100 * (1 - mf_rmse / bias_rmse)))
    print("Top-5 titles user 0:    %s" % model.recommend(0, n=5).tolist())
    print("MF beats both baselines:%s" % (mf_rmse < mean_rmse and mf_rmse < bias_rmse))
