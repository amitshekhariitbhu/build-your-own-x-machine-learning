import numpy as np

# Electricity Price Prediction Model (from scratch)
# -------------------------------------------------
# Hourly electricity prices are dominated by CALENDAR structure: a daily shape
# (cheap overnight, peaks morning/evening) and a weekly shape (weekends cheaper),
# on top of a slow trend and short-term AR persistence. We predict the next hour's
# price with a multiple linear regression solved by hand (ridge normal equations)
# on features that are all known at forecast time:
#   - autoregressive lags: price 1h, 24h and 168h (1 week) ago
#   - one-hot hour-of-day (24 cols) and a weekend flag (deterministic calendar)
# Beating the strong seasonal-naive baseline (price 24h ago) proves the model
# recovered persistence + trend + weekly effects the pure seasonal copy misses.


class ElectricityPricePredictor:
    """Ridge linear model on lag + calendar features; predicts next-hour price."""

    def __init__(self, lags=(1, 24, 168), ridge=1.0):
        self.lags = tuple(lags)
        self.ridge = ridge          # L2 penalty stabilizing the normal equations
        self.start = max(self.lags) # first index with full lag history
        self.w = None
        self.mu = self.sd = None    # standardization stats for the lag columns

    def _features(self, price, hour, weekend, ts):
        # Row per target index t, using ONLY info available at forecast time:
        # observed past prices (lags) and the deterministic calendar at t.
        ts = np.asarray(ts)
        lagcols = np.stack([price[ts - L] for L in self.lags], axis=1)  # AR features
        onehot = np.zeros((len(ts), 24))
        onehot[np.arange(len(ts)), hour[ts]] = 1.0                      # hour-of-day
        wknd = weekend[ts].reshape(-1, 1).astype(float)                 # weekend flag
        return lagcols, np.hstack([onehot, wknd])

    def fit(self, price, hour, weekend):
        ts = np.arange(self.start, len(price))
        lagcols, cal = self._features(price, hour, weekend, ts)
        self.mu, self.sd = lagcols.mean(0), lagcols.std(0) + 1e-12
        X = np.hstack([np.ones((len(ts), 1)), (lagcols - self.mu) / self.sd, cal])
        y = price[ts]
        A = X.T @ X
        reg = self.ridge * np.eye(X.shape[1]); reg[0, 0] = 0.0          # free the bias
        self.w = np.linalg.solve(A + reg, X.T @ y)
        return self

    def predict(self, price, hour, weekend, ts):
        lagcols, cal = self._features(price, hour, weekend, ts)
        X = np.hstack([np.ones((len(ts), 1)), (lagcols - self.mu) / self.sd, cal])
        return X @ self.w


def make_price_series(days=45, seed=0):
    # Synthetic hourly price with PLANTED daily + weekly seasonality, slow trend,
    # AR(1) persistence and noise -- structure the model can recover.
    np.random.seed(seed)
    n = days * 24
    t = np.arange(n)
    hour = (t % 24).astype(int)
    dow = ((t // 24) % 7).astype(int)          # 0=Mon .. 6=Sun
    weekend = (dow >= 5).astype(int)

    # deterministic daily shape: overnight trough, morning + evening peaks
    daily = (18 + 10 * np.sin((hour - 8) / 24 * 2 * np.pi)
             + 6 * np.sin((hour - 18) / 24 * 2 * np.pi))
    weekly = -6.0 * weekend                    # weekends noticeably cheaper
    trend = 0.02 * t                           # slow drift upward
    mean = daily + weekly + trend

    price = np.empty(n)
    price[0] = mean[0]
    resid = 0.0
    for i in range(1, n):                       # AR(1) residual around the mean shape
        resid = 0.6 * resid + np.random.normal(0, 1.5)
        price[i] = mean[i] + resid
    return np.maximum(price, 1.0), hour, weekend


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


if __name__ == "__main__":
    np.random.seed(0)
    price, hour, weekend = make_price_series(days=45)
    test_len = 168                              # forecast the held-out final week
    split = len(price) - test_len

    model = ElectricityPricePredictor(lags=(1, 24, 168), ridge=1.0)
    model.fit(price[:split], hour[:split], weekend[:split])

    # One-step-ahead over the held-out tail. Each row uses only true past prices
    # (lags reach back from t) and never peeks at price[t] itself.
    ts = np.arange(split, len(price))
    pred = model.predict(price, hour, weekend, ts)
    true = price[ts]

    naive = price[ts - 1]                        # last-value carry-forward
    seasonal = price[ts - 24]                    # seasonal-naive: price 24h ago
    mean_base = np.full(test_len, price[:split].mean())

    m_rmse, m_mae = rmse(pred, true), mae(pred, true)
    n_rmse = rmse(naive, true)
    s_rmse, s_mae = rmse(seasonal, true), mae(seasonal, true)
    b_rmse = rmse(mean_base, true)

    print("Electricity price prediction (ridge regression on lag + calendar features)")
    print(f"  hours={len(price)}  train={split}  test_tail={test_len}  features={len(model.w)}")
    print(f"  baseline mean            RMSE: {b_rmse:6.3f}")
    print(f"  baseline last-value      RMSE: {n_rmse:6.3f}")
    print(f"  baseline seasonal (24h)  RMSE: {s_rmse:6.3f}   MAE: {s_mae:6.3f}")
    print(f"  model    one-step        RMSE: {m_rmse:6.3f}   MAE: {m_mae:6.3f}")
    print(f"  RMSE improvement over seasonal-naive: {100 * (1 - m_rmse / s_rmse):.1f}%")
    print(f"  MAE  improvement over seasonal-naive: {100 * (1 - m_mae / s_mae):.1f}%")
    beat = m_rmse < s_rmse and m_rmse < n_rmse and m_rmse < b_rmse
    print(f"  result: {'model BEATS all baselines' if beat else 'baseline wins'}")
    print("  sample one-step forecasts vs truth (held-out week):")
    for j in range(0, test_len, test_len // 6):
        print(f"    hour={hour[ts][j]:2d}  pred={pred[j]:6.2f}  true={true[j]:6.2f}  seasonal={seasonal[j]:6.2f}")
