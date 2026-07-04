import numpy as np

# Rainfall Prediction Model (from scratch)
# ----------------------------------------
# Monthly rainfall is dominated by an ANNUAL cycle (wet/dry seasons) plus
# short-term persistence (a rainy month tends to follow a rainy month).
# We predict next-month rainfall with a linear model fit by least squares
# (solved via the normal equations, np.linalg.solve) over hand-built features:
#   [1, trend, sin(annual), cos(annual), rain_{t-1}, rain_{t-12}]
# The seasonal harmonics capture the yearly monsoon shape; the lags capture
# persistence and last-year's-same-month level. Rainfall is clipped at >= 0.


class RainfallPredictor:
    """One-step-ahead rainfall regressor on seasonal + lag features."""

    def __init__(self, season=12, ridge=1e-6):
        self.season = season
        self.ridge = ridge

    def _row(self, y, t):
        """Feature vector to predict y[t] from its own past."""
        ang = 2.0 * np.pi * (t % self.season) / self.season
        # trend normalized by the training length so fit/predict share one scale
        return [1.0, t / self.n_, np.sin(ang), np.cos(ang),
                y[t - 1], y[t - self.season]]

    def _design(self, y, idx):
        return np.asarray([self._row(y, t) for t in idx])

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        self.n_ = len(y)
        idx = np.arange(self.season, self.n_)          # need both lags available
        X = self._design(y, idx)
        target = y[idx]
        # normal equations w = (X'X + ridge*I)^-1 X'target
        XtX = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.w_ = np.linalg.solve(XtX, X.T @ target)
        return self

    def predict(self, y, idx):
        """Walk-forward one-step predictions at indices idx (uses true lags)."""
        X = self._design(np.asarray(y, dtype=float), idx)
        return np.maximum(0.0, X @ self.w_)            # rainfall is non-negative


def rmse(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def mae(a, b):
    return np.mean(np.abs(np.asarray(a) - np.asarray(b)))


if __name__ == "__main__":
    np.random.seed(0)

    # --- synthetic monthly rainfall (mm): annual monsoon cycle + trend + AR noise ---
    m = 12                                  # months per year
    n = 216                                 # 18 years
    t = np.arange(n)
    month = t % m
    # planted seasonal shape: strong wet peak mid-year, dry troughs
    seasonal = 60.0 + 55.0 * np.sin(2 * np.pi * (month - 3) / m) \
                     + 20.0 * np.cos(2 * np.pi * month / m)
    trend = 0.05 * t                        # slight long-term wetting
    rain = np.zeros(n)
    prev = 0.0
    for i in range(n):
        # AR(1) persistence around the seasonal+trend mean
        prev = 0.35 * prev + np.random.normal(0, 12.0)
        rain[i] = max(0.0, seasonal[i] + trend[i] + prev)

    # --- held-out tail: last 3 years ---
    h = 3 * m
    train_n = n - h
    test_idx = np.arange(train_n, n)
    test = rain[test_idx]

    # --- fit on the first 15 years, forecast the tail one step at a time ---
    model = RainfallPredictor(season=m).fit(rain[:train_n])
    pred = model.predict(rain, test_idx)

    # --- baselines ---
    naive = rain[test_idx - 1]              # last month's rainfall
    seasonal_naive = rain[test_idx - m]     # same month last year
    mean_base = np.full(h, rain[:train_n].mean())

    print("Rainfall Prediction Model vs baselines (held-out %d months)" % h)
    print("-" * 62)
    print("Mean baseline           RMSE=%6.3f  MAE=%6.3f" % (rmse(test, mean_base), mae(test, mean_base)))
    print("Naive (last month)      RMSE=%6.3f  MAE=%6.3f" % (rmse(test, naive), mae(test, naive)))
    print("Seasonal naive (t-12)   RMSE=%6.3f  MAE=%6.3f" % (rmse(test, seasonal_naive), mae(test, seasonal_naive)))
    print("Rainfall model (ours)   RMSE=%6.3f  MAE=%6.3f" % (rmse(test, pred), mae(test, pred)))
    print("-" * 62)

    best_base = min(rmse(test, mean_base), rmse(test, naive), rmse(test, seasonal_naive))
    ours = rmse(test, pred)
    improve = 100.0 * (best_base - ours) / best_base
    print("Best baseline RMSE=%6.3f -> model RMSE=%6.3f (%.1f%% lower)" % (best_base, ours, improve))
    print("RESULT:", "PASS - beats baseline" if ours < best_base else "FAIL")
