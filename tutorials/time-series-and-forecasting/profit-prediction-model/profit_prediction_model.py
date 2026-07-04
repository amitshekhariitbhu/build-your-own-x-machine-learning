import numpy as np

# Profit Prediction as time-series forecasting, from scratch.
# Monthly business profit is driven by a base LEVEL, a growth TREND, a 12-month
# SEASONAL pattern (holiday demand / summer slump), MARKETING spend (exogenous)
# with DIMINISHING returns -- doubling ad spend does not double profit, so the
# uplift is concave (~sqrt) -- and AR(1) momentum in the residual (a strong month
# carries into the next). We fit a linear regression on those features by ridge
# normal equations (solved by hand), then forecast a held-out tail of months.


def make_profit(T=132, seed=0):
    # Synthetic monthly profit ($k) with planted, recoverable structure.
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    month = t % 12
    # Planted 12-month seasonal shape: Q4 holiday surge, mid-year dip ($k).
    season = np.array([-6, -10, 2, 7, 11, 4, -8, -13, -3, 9, 24, 33], float)
    # Exogenous marketing spend ($k): baseline + occasional campaign bursts.
    mkt = 8.0 + 4.0 * rng.rand(T) + 10.0 * (rng.rand(T) < 0.20)
    level, growth, mkt_gain = 120.0, 0.55, 9.0
    resid = np.zeros(T)
    y = np.zeros(T)
    for k in range(T):
        eps = rng.normal(0, 4.0)
        resid[k] = (0.45 * resid[k - 1] if k else 0.0) + eps      # AR(1) momentum
        y[k] = (level + growth * t[k]              # base + rising trend
                + season[month[k]]                 # seasonal pattern
                + mkt_gain * np.sqrt(mkt[k])       # concave marketing uplift
                + resid[k])                        # correlated shock
    return y, mkt, month


class ProfitForecaster:
    """Regression forecaster:
    y_t ~ c + a*t + season(month) + phi*y_{t-1} + b*sqrt(mkt_t) + d*mkt_t."""

    def __init__(self, lam=1e-2):
        self.lam = lam

    def _design(self, y, mkt, month, idx):
        t = idx.astype(float)
        # Month one-hot with 11 dummies (Jan is the reference baseline).
        oh = np.eye(12)[month[idx]][:, 1:]
        return np.column_stack([
            np.ones_like(t), t, oh,
            y[idx - 1],                            # last month's profit (AR term)
            np.sqrt(mkt[idx]), mkt[idx],           # concave + linear marketing
        ])

    def fit(self, y, mkt, month, idx):
        Z = self._design(y, mkt, month, idx)
        R = self.lam * np.eye(Z.shape[1])
        R[0, 0] = 0.0                              # don't penalize the intercept
        self.w = np.linalg.solve(Z.T @ Z + R, Z.T @ y[idx])
        return self

    def predict(self, y, mkt, month, idx):
        # One-step-ahead: uses the TRUE previous month and known marketing spend.
        return self._design(y, mkt, month, idx) @ self.w

    def forecast(self, y, mkt, month, start, steps):
        # Recursive multi-step: feed each predicted month back in as next y_{t-1}.
        hist = y.astype(float).copy()
        out = []
        for h in range(steps):
            k = start + h
            oh = np.eye(12)[month[k]][1:]
            feat = np.concatenate([[1.0, float(k)], oh,
                                   [hist[k - 1], np.sqrt(mkt[k]), mkt[k]]])
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

    y, mkt, month = make_profit(T=132, seed=0)
    split = 108                                    # held-out tail = last 24 months
    tr = np.arange(12, split)                      # start at 12 (align with seasonal)
    te = np.arange(split, len(y))

    model = ProfitForecaster(lam=1e-2).fit(y, mkt, month, tr)

    pred = model.predict(y, mkt, month, te)        # one-step-ahead forecasts
    naive = y[te - 1]                              # persistence (last month) baseline
    seasonal = y[te - 12]                          # seasonal-naive (last year) baseline
    mean_base = np.full(len(te), y[tr].mean())     # predict-mean baseline
    true = y[te]

    print("Profit series: %d months   train=%d  test=%d   ($k/month)"
          % (len(y), len(tr), len(te)))
    print("-" * 62)
    print("Regression (trend+season+lag+mkt) RMSE: %7.3f  MAE: %7.3f"
          % (rmse(pred, true), mae(pred, true)))
    print("Seasonal-naive (y_{t-12}) baseline RMSE: %7.3f  MAE: %7.3f"
          % (rmse(seasonal, true), mae(seasonal, true)))
    print("Naive last-month baseline          RMSE: %7.3f  MAE: %7.3f"
          % (rmse(naive, true), mae(naive, true)))
    print("Predict-mean baseline              RMSE: %7.3f  MAE: %7.3f"
          % (rmse(mean_base, true), mae(mean_base, true)))
    print("-" * 62)

    fc = model.forecast(y, mkt, month, split, 6)
    print("Recursive 6-month profit forecast from train end:")
    for i, (p, tval) in enumerate(zip(fc, true[:6]), 1):
        print("   month+%d  pred=%8.2f   true=%8.2f" % (i, p, tval))
    print("-" * 62)
    beats = rmse(pred, true) < min(rmse(naive, true),
                                   rmse(seasonal, true),
                                   rmse(mean_base, true))
    print("Regression beats naive, seasonal & mean baselines: %s" % beats)
