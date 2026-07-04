import numpy as np

class DailyBirthForecaster:
    """One-step forecaster for daily birth counts. Features per day:
    recent AR lags + a linear trend + day-of-week one-hot (weekly seasonality).
    Fit from scratch via ridge normal equations (no libraries)."""
    def __init__(self, n_lags=7, ridge=1e-2):
        self.n_lags = n_lags        # autoregressive order: how many past days feed a forecast
        self.ridge = ridge          # L2 penalty stabilizes the normal equations
        self.w = None

    def _features(self, series, times):
        # Row per target day `t`: [lag_1..lag_p, bias, trend, dow_0..dow_6].
        times = np.atleast_1d(times).astype(int)
        lags = np.stack([series[times - k] for k in range(1, self.n_lags + 1)], axis=1)
        bias = np.ones((len(times), 1))
        trend = (times / 7.0).reshape(-1, 1)          # slow drift, scaled by week
        dow = np.zeros((len(times), 7))               # weekly seasonal dummies
        dow[np.arange(len(times)), times % 7] = 1.0
        return np.hstack([lags, bias, trend, dow])

    def fit(self, series):
        # Train on every day whose full lag window exists.
        series = np.asarray(series, float)
        self._trainlen = len(series)
        times = np.arange(self.n_lags, len(series))
        X = self._features(series, times)
        y = series[times]
        # Ridge: w = (X'X + lambda*I)^-1 X'y, but don't penalize the bias column.
        A = X.T @ X
        reg = self.ridge * np.eye(X.shape[1]); reg[self.n_lags, self.n_lags] = 0.0
        self.w = np.linalg.solve(A + reg, X.T @ y)
        return self

    def predict_onestep(self, series, times):
        # One-step-ahead using TRUE past days (times index into `series`).
        return self._features(np.asarray(series, float), times) @ self.w

    def forecast(self, history, steps):
        # Pure multi-step: recurse, feeding predictions back in as lags.
        buf = list(np.asarray(history, float))
        t = self._trainlen
        out = []
        for _ in range(steps):
            lags = [buf[-k] for k in range(1, self.n_lags + 1)]
            bias, trend = 1.0, t / 7.0
            dow = np.zeros(7); dow[t % 7] = 1.0
            row = np.hstack([lags, bias, trend, dow])
            yhat = float(row @ self.w)
            out.append(yhat); buf.append(yhat); t += 1
        return np.array(out)

def make_births(n=365, seed=0):
    # Synthetic daily birth counts: base level + slow trend + weekly cycle + AR(1) noise.
    # Latent structure to recover: fewer births on weekends, gentle yearly rise.
    np.random.seed(seed)
    t = np.arange(n)
    base = 42.0 + 0.015 * t                                    # rising baseline
    weekly = -6.0 * (t % 7 >= 5) + 2.0 * np.sin(2 * np.pi * t / 7)  # weekend dip
    noise = np.zeros(n)
    for i in range(1, n):                                      # AR(1) noise -> lags matter
        noise[i] = 0.55 * noise[i - 1] + np.random.randn() * 1.5
    births = base + weekly + noise
    return np.maximum(births, 0).round()                       # counts are non-negative ints

if __name__ == "__main__":
    np.random.seed(0)
    test_len = 60
    series = make_births(n=365)
    split = len(series) - test_len                            # forecast the held-out tail

    model = DailyBirthForecaster(n_lags=7, ridge=1e-2)
    model.fit(series[:split])

    idx = np.arange(split, len(series))
    pred = model.predict_onestep(series, idx)
    true = series[idx]
    naive = series[idx - 1]                                   # baseline: yesterday's count
    seasonal = series[idx - 7]                                # baseline: same weekday last week
    mean_base = np.full_like(true, series[:split].mean())     # baseline: train mean

    multi = model.forecast(series[:split], test_len)          # pure recursive forecast

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, n_rmse = rmse(pred, true), rmse(naive, true)
    s_rmse, mn_rmse = rmse(seasonal, true), rmse(mean_base, true)
    m_mae, n_mae = mae(pred, true), mae(naive, true)
    multi_rmse = rmse(multi, true)
    best_base = min(n_rmse, s_rmse, mn_rmse)

    print("Daily birth forecasting (AR lags + trend + weekly seasonality, ridge least squares)")
    print(f"  days={len(series)}  lags={model.n_lags}  test_tail={test_len}")
    print(f"  baseline mean          RMSE: {mn_rmse:.3f}")
    print(f"  baseline last-value    RMSE: {n_rmse:.3f}   MAE: {n_mae:.3f}")
    print(f"  baseline seasonal(-7d) RMSE: {s_rmse:.3f}")
    print(f"  model one-step         RMSE: {m_rmse:.3f}   MAE: {m_mae:.3f}")
    print(f"  model multi-step (recursive) RMSE: {multi_rmse:.3f}")
    print(f"  one-step RMSE improvement over best baseline: {100*(1-m_rmse/best_base):.1f}%")
    print(f"  one-step MAE  improvement over last-value:    {100*(1-m_mae/n_mae):.1f}%")
    print(f"  result: {'model BEATS all naive baselines' if m_rmse < best_base else 'baseline wins'}")
    print("  sample one-step forecasts vs truth (held-out tail):")
    for j in range(0, len(true), max(1, len(true) // 6)):
        print(f"    pred={pred[j]:6.1f}   true={true[j]:6.1f}   naive={naive[j]:6.1f}")
