import numpy as np


class FollowerPredictor:
    """Ridge-regularized linear regressor for follower counts, from scratch.

    Follower counts are heavy-tailed, so the model predicts log1p(followers)
    from standardized account features via full-batch gradient descent on the
    mean-squared-error loss (+ L2). predict() maps back with expm1.
    """

    def __init__(self, lr=0.1, n_iter=4000, l2=1e-2):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # L2 (ridge) strength
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    def _standardize(self, X):
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.log1p(np.asarray(y, dtype=float).reshape(-1))  # model log-followers
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = y.mean()
        for _ in range(self.n_iter):
            pred = Xs @ self.w + self.b            # predicted log-followers
            err = pred - y
            grad_w = Xs.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, X):
        Xs = self._standardize(np.asarray(X, dtype=float))
        return np.expm1(Xs @ self.w + self.b)      # back to raw follower counts


def make_account_data(n=1500):
    """Synthetic social accounts with a planted follower-growth rule.

    Features per account:
      account_age_mo : months since the account was created
      posts_per_week : posting frequency
      engagement_rate: avg (likes+comments)/reach, 0..1
      hashtags_avg   : hashtags used per post
      verified       : blue-check badge (0/1)
      collab_count   : cross-promotions / collabs with other accounts
    Followers grow with age, engagement, verification and collabs; extreme
    hashtag stuffing hurts. Built in log-space, then exponentiated + noised.
    """
    account_age_mo = np.random.gamma(4.0, 6.0, n)
    posts_per_week = np.random.gamma(2.0, 3.0, n)
    engagement_rate = np.clip(np.random.beta(2, 8, n), 0, 1)
    hashtags_avg = np.random.randint(0, 30, n).astype(float)
    verified = (np.random.rand(n) > 0.85).astype(float)
    collab_count = np.random.poisson(3, n).astype(float)
    X = np.column_stack([account_age_mo, posts_per_week, engagement_rate,
                         hashtags_avg, verified, collab_count])

    # planted log-follower signal (diminishing returns via logs / sqrt)
    log_f = (4.0
             + 0.9 * np.log1p(account_age_mo)
             + 0.5 * np.log1p(posts_per_week)
             + 6.0 * engagement_rate
             - 0.02 * hashtags_avg
             + 1.3 * verified
             + 0.25 * collab_count)
    log_f += np.random.normal(0, 0.35, n)          # organic variability
    followers = np.expm1(log_f)
    return X, followers


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_account_data(1500)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = FollowerPredictor(lr=0.1, n_iter=4000, l2=1e-2).fit(Xtr, ytr)
    pred = model.predict(Xte)

    # metrics on raw follower counts; baseline = predict the training mean
    base = np.full_like(yte, ytr.mean(), dtype=float)
    rmse = np.sqrt(np.mean((pred - yte) ** 2))
    base_rmse = np.sqrt(np.mean((base - yte) ** 2))
    mae = np.mean(np.abs(pred - yte))
    base_mae = np.mean(np.abs(base - yte))
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    feats = ["age_mo", "posts_wk", "engage", "hashtags", "verified", "collabs"]
    print("test accounts         :", len(yte))
    print("mean followers (test) :", int(yte.mean()))
    print("baseline RMSE (mean)  :", int(base_rmse))
    print("model RMSE            :", int(rmse))
    print("baseline MAE (mean)   :", int(base_mae))
    print("model MAE             :", int(mae))
    print("R^2 (0=mean,1=perfect):", round(float(r2), 3))
    print("learned weights       :", dict(zip(feats, np.round(model.w, 2))))
    print("BEATS baseline        :", bool(rmse < base_rmse and r2 > 0.0))
