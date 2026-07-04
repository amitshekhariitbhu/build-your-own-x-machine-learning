import numpy as np


class GameSalesRegressor:
    """Ridge-regularized linear regressor for video-game global sales, from scratch.

    Unit sales are heavy-tailed (a few blockbusters, many niche titles), so the
    model predicts log1p(sales_millions) from standardized game features via
    full-batch gradient descent on the MSE loss (+ L2). predict() maps back with
    expm1 to return sales in millions of units.
    """

    def __init__(self, lr=0.1, n_iter=5000, l2=1e-2):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2            # L2 (ridge) strength
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    def _standardize(self, X):
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.log1p(np.asarray(y, dtype=float).reshape(-1))   # model log-sales
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = y.mean()
        for _ in range(self.n_iter):
            pred = Xs @ self.w + self.b            # predicted log-sales
            err = pred - y
            grad_w = Xs.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, X):
        Xs = self._standardize(np.asarray(X, dtype=float))
        return np.expm1(Xs @ self.w + self.b)     # back to sales in millions


def make_game_data(n=1600):
    """Synthetic game catalog with a planted sales rule.

    Features per title:
      critic_score      : aggregated review score, 0..100
      user_score        : player rating, 0..10
      marketing_usd_m   : marketing spend in $M
      platform_base_m   : install base of target console (millions)
      is_franchise      : part of an established franchise / sequel (0/1)
      num_platforms     : how many platforms it launched on
      genre_appeal      : mass-market pull of the genre, 0..1
    Sales rise with reviews, marketing, install base, franchise pull, wider
    platform coverage and genre appeal. Built in log-space, then exponentiated
    and noised to reproduce the blockbuster-heavy long tail.
    """
    critic_score = np.clip(np.random.normal(70, 12, n), 30, 99)
    user_score = np.clip(np.random.normal(7.0, 1.3, n), 1, 10)
    marketing_usd_m = np.random.gamma(2.0, 8.0, n)
    platform_base_m = np.random.gamma(3.0, 25.0, n)
    is_franchise = (np.random.rand(n) > 0.7).astype(float)
    num_platforms = np.random.randint(1, 5, n).astype(float)
    genre_appeal = np.clip(np.random.beta(2, 3, n), 0, 1)
    X = np.column_stack([critic_score, user_score, marketing_usd_m,
                         platform_base_m, is_franchise, num_platforms,
                         genre_appeal])

    # planted log-sales signal (diminishing returns via logs)
    log_s = (-8.5
             + 0.045 * critic_score
             + 0.20 * user_score
             + 0.55 * np.log1p(marketing_usd_m)
             + 0.60 * np.log1p(platform_base_m)
             + 0.85 * is_franchise
             + 0.30 * num_platforms
             + 1.40 * genre_appeal)
    log_s += np.random.normal(0, 0.30, n)          # market variability
    sales = np.expm1(log_s)                         # millions of units
    return X, sales


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_game_data(1600)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = GameSalesRegressor(lr=0.1, n_iter=5000, l2=1e-2).fit(Xtr, ytr)
    pred = model.predict(Xte)

    # metrics on raw sales (millions); baseline = predict the training mean
    base = np.full_like(yte, ytr.mean(), dtype=float)
    rmse = np.sqrt(np.mean((pred - yte) ** 2))
    base_rmse = np.sqrt(np.mean((base - yte) ** 2))
    mae = np.mean(np.abs(pred - yte))
    base_mae = np.mean(np.abs(base - yte))
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    feats = ["critic", "user", "mktg", "base", "franch", "nplat", "genre"]
    print("test titles           :", len(yte))
    print("mean sales M (test)   :", round(float(yte.mean()), 3))
    print("baseline RMSE (mean)  :", round(float(base_rmse), 3))
    print("model RMSE            :", round(float(rmse), 3))
    print("baseline MAE (mean)   :", round(float(base_mae), 3))
    print("model MAE             :", round(float(mae), 3))
    print("R^2 (0=mean,1=perfect):", round(float(r2), 3))
    print("learned weights       :", dict(zip(feats, np.round(model.w, 2))))
    print("BEATS baseline        :", bool(rmse < base_rmse and r2 > 0.0))
