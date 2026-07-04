import numpy as np

class ARForecaster:
    """Univariate one-step forecaster: ridge least-squares on lagged values + a
    deterministic trend/seasonal basis. Truly from scratch (normal equations)."""
    def __init__(self, n_lags=8, period=25, n_harmonics=2, ridge=1e-3):
        self.n_lags = n_lags          # autoregressive order (past values as features)
        self.period = period          # known seasonal cycle length
        self.n_harmonics = n_harmonics
        self.ridge = ridge            # L2 penalty stabilizes the normal equations
        self.w = None

    def _det_features(self, t):
        # Deterministic basis at absolute time index t: bias, linear trend, Fourier terms.
        t = np.atleast_1d(t).astype(float)
        feats = [np.ones_like(t), t / self.period]
        for k in range(1, self.n_harmonics + 1):
            feats.append(np.sin(2 * np.pi * k * t / self.period))
            feats.append(np.cos(2 * np.pi * k * t / self.period))
        return np.stack(feats, axis=1)   # (len(t), 2 + 2*H)

    def _design(self, series, times):
        # Row for each target: [lag_1..lag_p, deterministic(t)]. `times` = target indices.
        L = np.stack([series[times - k] for k in range(1, self.n_lags + 1)], axis=1)
        return np.hstack([L, self._det_features(times)])

    def fit(self, series):
        # Train on every point whose full lag window exists.
        series = np.asarray(series, float)
        self._trainlen = len(series)
        times = np.arange(self.n_lags, len(series))
        X = self._design(series, times)
        y = series[times]
        # Ridge normal equations: w = (X'X + λI)^-1 X'y  (don't penalize the bias col).
        A = X.T @ X
        reg = self.ridge * np.eye(X.shape[1]); reg[self.n_lags, self.n_lags] = 0.0
        self.w = np.linalg.solve(A + reg, X.T @ y)
        return self

    def predict_onestep(self, series, times):
        # One-step-ahead using TRUE history (times index into `series`).
        return self._design(np.asarray(series, float), np.asarray(times)) @ self.w

    def forecast(self, history, steps):
        # Recursive multi-step: feed predictions back in as lags.
        buf = list(np.asarray(history, float))
        t = self._trainlen
        out = []
        for _ in range(steps):
            lags = np.array([buf[-k] for k in range(1, self.n_lags + 1)])
            row = np.hstack([lags, self._det_features(t)[0]])
            yhat = float(row @ self.w)
            out.append(yhat); buf.append(yhat); t += 1
        return np.array(out)

def make_series(n=400, period=25, seed=0):
    # Trend + two seasonal cycles + AR(1)-correlated noise: recoverable latent structure.
    np.random.seed(seed)
    t = np.arange(n)
    trend = 0.02 * t
    seasonal = 3.0 * np.sin(2 * np.pi * t / period) + 1.0 * np.sin(2 * np.pi * t / 7)
    noise = np.zeros(n)
    for i in range(1, n):                       # AR(1) noise -> lags carry real signal
        noise[i] = 0.5 * noise[i - 1] + np.random.randn() * 0.4
    return trend + seasonal + noise

if __name__ == "__main__":
    np.random.seed(0)
    period, test_len = 25, 60
    series = make_series(n=400, period=period)
    split = len(series) - test_len              # forecast the held-out tail

    model = ARForecaster(n_lags=8, period=period, n_harmonics=2, ridge=1e-3)
    model.fit(series[:split])

    # One-step-ahead over the held-out tail (each forecast uses only true past values).
    idx = np.arange(split, len(series))
    pred = model.predict_onestep(series, idx)
    true = series[idx]
    naive = series[idx - 1]                     # baseline: predict the previous value
    mean_base = np.full_like(true, series[:split].mean())  # baseline: train mean

    # Pure multi-step forecast (no peeking): recurse from the end of training.
    multi = model.forecast(series[:split], test_len)

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, n_rmse, mn_rmse = rmse(pred, true), rmse(naive, true), rmse(mean_base, true)
    m_mae, n_mae = mae(pred, true), mae(naive, true)
    multi_rmse = rmse(multi, true)

    print("Univariate time-series forecasting (AR + trend/seasonal, ridge least squares)")
    print(f"  series={len(series)}  lags={model.n_lags}  period={period}  test_tail={test_len}")
    print(f"  baseline mean       RMSE: {mn_rmse:.3f}")
    print(f"  baseline last-value RMSE: {n_rmse:.3f}   MAE: {n_mae:.3f}")
    print(f"  AR one-step         RMSE: {m_rmse:.3f}   MAE: {m_mae:.3f}")
    print(f"  AR multi-step (recursive) RMSE: {multi_rmse:.3f}")
    print(f"  one-step RMSE improvement over last-value: {100*(1-m_rmse/n_rmse):.1f}%")
    print(f"  one-step MAE  improvement over last-value: {100*(1-m_mae/n_mae):.1f}%")
    print(f"  result: {'AR model BEATS naive baselines' if m_rmse < min(n_rmse, mn_rmse) else 'baseline wins'}")
    print("  sample one-step forecasts vs truth (held-out tail):")
    for j in range(0, len(true), max(1, len(true) // 6)):
        print(f"    pred={pred[j]:7.3f}   true={true[j]:7.3f}   naive={naive[j]:7.3f}")
