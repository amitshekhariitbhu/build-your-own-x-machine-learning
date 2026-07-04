import numpy as np

# Future Sales Prediction as time-series forecasting, from scratch.
# Monthly store sales are driven by a base LEVEL, a growth TREND, a 12-month
# SEASONAL pattern (holiday peaks / summer lulls), PROMOTION bursts (exogenous),
# and AR(1) momentum in the residual (a good/bad month spills into the next).
# We fit a linear regression on those features -- trend, seasonal dummies, the
# last month's sales (y_{t-1}), the same-month-last-year sales (y_{t-12}), and
# the promo flag -- by ridge normal equations, then forecast a held-out tail.


def make_sales(T=132, seed=0):
    # Synthetic monthly sales with planted, recoverable structure (units sold).
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    month = t % 12
    # Planted 12-month seasonal shape: Q4 holiday spike, mid-year dip.
    season_shape = np.array([-8, -14, 4, 10, 16, 6, -10, -18, -4, 12, 34, 48], float)
    promo = (rng.rand(T) < 0.18).astype(float)     # exogenous promo months
    level, growth, promo_gain = 200.0, 0.85, 42.0
    resid = np.zeros(T)
    y = np.zeros(T)
    for k in range(T):
        eps = rng.normal(0, 6.0)
        resid[k] = (0.5 * resid[k - 1] if k else 0.0) + eps   # AR(1) momentum
        y[k] = (level + growth * t[k]              # base + rising trend
                + season_shape[month[k]]           # seasonal pattern
                + promo_gain * promo[k]            # promotion uplift
                + resid[k])                        # correlated shock
    return y, promo, month


class SalesForecaster:
    """Regression forecaster: y_t ~ c + a*t + season(month) + phi*y_{t-1} + s*y_{t-12} + g*promo."""

    def __init__(self, lam=1e-2):
        self.lam = lam

    def _design(self, y, promo, month, idx):
        t = idx.astype(float)
        # Month one-hot with 11 dummies (Jan is the reference baseline).
        oh = np.eye(12)[month[idx]][:, 1:]
        return np.column_stack([
            np.ones_like(t), t, oh,
            y[idx - 1], y[idx - 12], promo[idx],
        ])

    def fit(self, y, promo, month, idx):
        Z = self._design(y, promo, month, idx)
        R = self.lam * np.eye(Z.shape[1])
        R[0, 0] = 0.0                              # don't penalize the intercept
        self.w = np.linalg.solve(Z.T @ Z + R, Z.T @ y[idx])
        return self

    def predict(self, y, promo, month, idx):
        # One-step-ahead: uses the TRUE previous months and known promo flag.
        return self._design(y, promo, month, idx) @ self.w

    def forecast(self, y, promo, month, start, steps):
        # Recursive multi-step: feed each predicted month back in as next y_{t-1}.
        hist = y.astype(float).copy()
        out = []
        for h in range(steps):
            k = start + h
            oh = np.eye(12)[month[k]][1:]
            feat = np.concatenate([[1.0, float(k)], oh,
                                   [hist[k - 1], hist[k - 12], promo[k]]])
            yp = feat @ self.w
            hist[k] = yp
            out.append(yp)
        return np.array(out)


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def mae(a, b):
    return np.mean(np.abs(a - b))


if __name__ == "__main__":
    np.random.seed(0)

    y, promo, month = make_sales(T=132, seed=0)
    split = 108                                    # held-out tail = last 24 months
    tr = np.arange(12, split)                      # start at 12 (needs y_{t-12})
    te = np.arange(split, len(y))

    model = SalesForecaster(lam=1e-2).fit(y, promo, month, tr)

    pred = model.predict(y, promo, month, te)      # one-step-ahead forecasts
    naive = y[te - 1]                              # persistence (last month) baseline
    seasonal = y[te - 12]                          # seasonal-naive (last year) baseline
    mean_base = np.full(len(te), y[tr].mean())     # predict-mean baseline
    true = y[te]

    print("Sales series: %d months   train=%d  test=%d   (units/month)"
          % (len(y), len(tr), len(te)))
    print("-" * 62)
    print("Regression (trend+season+lags+promo) RMSE: %7.3f  MAE: %7.3f"
          % (rmse(pred, true), mae(pred, true)))
    print("Seasonal-naive  (y_{t-12}) baseline  RMSE: %7.3f  MAE: %7.3f"
          % (rmse(seasonal, true), mae(seasonal, true)))
    print("Naive last-month baseline            RMSE: %7.3f  MAE: %7.3f"
          % (rmse(naive, true), mae(naive, true)))
    print("Predict-mean baseline                RMSE: %7.3f  MAE: %7.3f"
          % (rmse(mean_base, true), mae(mean_base, true)))
    print("-" * 62)

    fc = model.forecast(y, promo, month, split, 6)
    print("Recursive 6-month forecast from train end:")
    for i, (p, tval) in enumerate(zip(fc, true[:6]), 1):
        print("   month+%d  pred=%8.2f   true=%8.2f" % (i, p, tval))
    print("-" * 62)
    beats = rmse(pred, true) < min(rmse(naive, true),
                                   rmse(seasonal, true),
                                   rmse(mean_base, true))
    print("Regression beats naive, seasonal & mean baselines: %s" % beats)
