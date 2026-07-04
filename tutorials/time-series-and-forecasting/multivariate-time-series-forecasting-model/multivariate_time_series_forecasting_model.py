import numpy as np

# Multivariate time series forecasting via Vector AutoRegression (VAR), from scratch.
# Each variable is predicted from the recent lags of ALL variables, so cross-series
# coupling is exploited. Coefficients are fit by (ridge) least squares / normal eqs.


def _companion(As):
    # Companion matrix of a VAR(p): its spectral radius < 1 => stationary process.
    k, p = As[0].shape[0], len(As)
    C = np.zeros((k * p, k * p))
    C[:k] = np.hstack(As)
    if p > 1:
        C[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    return C


def make_series(T=600, seed=0):
    # Synthetic 3-variable series from a stable VAR(2) with planted CROSS-coupling:
    # off-diagonal terms make each variable depend on the others' past -> the
    # multivariate structure is real and recoverable, and univariate models miss it.
    rng = np.random.RandomState(seed)
    A1 = np.array([[0.40, 0.32, 0.00],
                   [0.00, 0.35, 0.34],
                   [0.28, 0.00, 0.38]])
    A2 = np.array([[0.10, 0.00, 0.16],
                   [0.12, 0.00, 0.00],
                   [0.00, 0.22, 0.00]])
    while max(abs(np.linalg.eigvals(_companion([A1, A2])))) >= 0.92:
        A1, A2 = A1 * 0.9, A2 * 0.9          # shrink until stationary
    c = np.array([0.5, -0.3, 0.2])           # drift -> nonzero level
    k = 3
    X = np.zeros((T, k))
    X[:2] = rng.normal(0, 1, (2, k))
    for t in range(2, T):
        X[t] = c + A1 @ X[t - 1] + A2 @ X[t - 2] + rng.normal(0, 0.5, k)
    return X, (A1, A2)


class VAR:
    """Vector AutoRegression: X_t ~ c + A1 X_{t-1} + ... + Ap X_{t-p}."""

    def __init__(self, p=2, lam=1e-3):
        self.p, self.lam = p, lam

    def _design(self, X):
        # Row t holds [1, X_{t-1}, X_{t-2}, ..., X_{t-p}] (most-recent lag first).
        T, k, p = X.shape[0], X.shape[1], self.p
        Z = np.array([X[t - p:t][::-1].reshape(-1) for t in range(p, T)])
        Z = np.hstack([np.ones((len(Z), 1)), Z])
        return Z, X[p:]                       # design (T-p, k*p+1), targets (T-p, k)

    def fit(self, X):
        Z, Y = self._design(X)
        R = self.lam * np.eye(Z.shape[1])
        R[0, 0] = 0.0                         # do not penalize the intercept
        self.B = np.linalg.solve(Z.T @ Z + R, Z.T @ Y)   # (k*p+1, k)
        return self

    def _feat(self, hist):
        # hist: last p rows in chronological order (oldest first).
        return np.concatenate([[1.0], hist[::-1].reshape(-1)])

    def predict_one(self, hist):
        return self._feat(hist[-self.p:]) @ self.B

    def forecast(self, hist, steps):
        # Recursive multi-step: feed each prediction back in as the newest lag.
        h = list(hist[-self.p:])
        out = []
        for _ in range(steps):
            x = self._feat(np.array(h[-self.p:])) @ self.B
            out.append(x)
            h.append(x)
        return np.array(out)


def fit_univariate(X, p):
    # Independent AR(p) per series (ignores cross-coupling) -> the multivariate baseline.
    T, k = X.shape
    models = []
    for j in range(k):
        Z = np.array([X[t - p:t, j][::-1] for t in range(p, T)])
        Z = np.hstack([np.ones((len(Z), 1)), Z])
        w = np.linalg.solve(Z.T @ Z + 1e-3 * np.eye(Z.shape[1]), Z.T @ X[p:, j])
        models.append(w)
    return models


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


if __name__ == "__main__":
    np.random.seed(0)

    X, _ = make_series(T=600, seed=0)
    p = 2
    split = 480                               # held-out tail = last 120 steps
    Xtr, Xte = X[:split], X[split:]

    model = VAR(p=p, lam=1e-3).fit(Xtr)
    uni = fit_univariate(Xtr, p)

    # One-step-ahead forecasts over the held-out tail using true past lags.
    steps = range(split, len(X))
    var_pred = np.array([model.predict_one(X[t - p:t]) for t in steps])
    uni_pred = np.array([[np.concatenate([[1.0], X[t - p:t, j][::-1]]) @ uni[j]
                          for j in range(X.shape[1])] for t in steps])
    naive = X[split - 1:len(X) - 1]           # last-value (persistence) baseline
    mean_base = np.tile(Xtr.mean(0), (len(Xte), 1))
    true = X[split:]

    r_var, r_uni = rmse(var_pred, true), rmse(uni_pred, true)
    r_naive, r_mean = rmse(naive, true), rmse(mean_base, true)

    print("Multivariate series: %d steps x %d variables   train=%d  test=%d  lags=%d"
          % (X.shape[0], X.shape[1], split, len(Xte), p))
    print("-" * 66)
    print("VAR (multivariate)     one-step RMSE: %.4f   MAE: %.4f"
          % (r_var, np.mean(np.abs(var_pred - true))))
    print("Univariate AR (per-series)  RMSE: %.4f   <- ignores cross-coupling" % r_uni)
    print("Naive last-value baseline   RMSE: %.4f" % r_naive)
    print("Predict-mean baseline       RMSE: %.4f" % r_mean)
    print("-" * 66)
    print("Per-variable VAR RMSE: " + "  ".join("v%d=%.3f" % (j, rmse(var_pred[:, j], true[:, j]))
                                                 for j in range(X.shape[1])))
    fc = model.forecast(Xtr, 3)               # sample recursive multi-step forecast
    print("Recursive 3-step forecast from train end: ")
    for i, row in enumerate(fc, 1):
        print("   t+%d  pred=[% .3f % .3f % .3f]   true=[% .3f % .3f % .3f]"
              % (i, row[0], row[1], row[2], X[split - 1 + i, 0], X[split - 1 + i, 1], X[split - 1 + i, 2]))
    print("-" * 66)
    print("VAR beats naive & mean & univariate: %s"
          % (r_var < r_naive and r_var < r_mean and r_var < r_uni))
