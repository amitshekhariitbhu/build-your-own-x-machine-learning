import numpy as np

class ProphetForecaster:
    """Prophet-style additive model: y(t) = piecewise-linear trend + Fourier seasonality.

    Trend is g(t) = k*t + sum_j delta_j * (t - cp_j)_+  (continuous piecewise-linear,
    changepoints auto-placed over early history). Seasonality is a Fourier series per
    period. All coefficients are fit jointly by ridge regression, with an L2 penalty on
    the changepoint deltas (mimicking Prophet's sparse changepoint prior) so the trend
    stays smooth and extrapolates cleanly.
    """
    def __init__(self, periods=((7, 3), (30, 4)), n_changepoints=20,
                 changepoint_range=0.8, cp_prior=1.0, season_prior=10.0):
        self.periods = periods            # list of (period_in_days, fourier_order)
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.cp_prior = cp_prior          # smaller -> stiffer trend (more regularized deltas)
        self.season_prior = season_prior

    def _features(self, t):
        # t: day indices (float). Trend uses time scaled by the training horizon.
        ts = t / self.t_scale
        cols = [np.ones_like(ts), ts]                     # bias, linear slope
        reg = [0.0, 0.0]                                  # no penalty on base trend
        for cp in self.changepoints:                      # hinge per changepoint
            cols.append(np.maximum(0.0, ts - cp))
            reg.append(1.0 / self.cp_prior)               # penalize delta_j
        for P, K in self.periods:                         # Fourier seasonality
            for k in range(1, K + 1):
                cols.append(np.sin(2 * np.pi * k * t / P))
                cols.append(np.cos(2 * np.pi * k * t / P))
                reg += [1.0 / self.season_prior] * 2
        return np.column_stack(cols), np.asarray(reg)

    def fit(self, t, y):
        t = np.asarray(t, float)
        self.t_scale = t.max() - t.min() + 1e-9
        # Changepoints on scaled time, uniformly over the first `changepoint_range`.
        hi = self.changepoint_range * (t.max() - t.min()) / self.t_scale
        self.changepoints = np.linspace(0, hi, self.n_changepoints + 1)[1:]
        X, reg = self._features(t)
        # Ridge normal equations: (X^T X + diag(reg^2)) theta = X^T y  (closed form).
        A = X.T @ X + np.diag(reg ** 2)
        self.theta = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, t):
        X, _ = self._features(np.asarray(t, float))
        return X @ self.theta


def make_series(n=360, seed=0):
    # Synthetic daily series: piecewise-linear trend (slope change at t=180) +
    # weekly and monthly seasonality + mild noise -> exactly the structure Prophet recovers.
    np.random.seed(seed)
    t = np.arange(n)
    trend = np.where(t < 180, 0.05 * t, 0.05 * 180 - 0.03 * (t - 180))
    weekly = 2.5 * np.sin(2 * np.pi * t / 7)
    monthly = 4.0 * np.sin(2 * np.pi * t / 30 + 0.7)
    noise = np.random.randn(n) * 0.6
    return t.astype(float), 20.0 + trend + weekly + monthly + noise


if __name__ == "__main__":
    np.random.seed(0)
    t, y = make_series()
    test_len = 45
    split = len(t) - test_len                     # forecast the held-out tail
    tr_t, tr_y, te_t, te_y = t[:split], y[:split], t[split:], y[split:]

    model = ProphetForecaster().fit(tr_t, tr_y)
    pred = model.predict(te_t)                    # multi-step forecast over unseen tail

    naive = np.full(test_len, tr_y[-1])           # baseline: last observed value
    mean_base = np.full(test_len, tr_y.mean())    # baseline: training mean

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    p_rmse, p_mae = rmse(pred, te_y), mae(pred, te_y)
    n_rmse, n_mae = rmse(naive, te_y), mae(naive, te_y)
    m_rmse = rmse(mean_base, te_y)

    print("Prophet-style forecasting (piecewise-linear trend + Fourier seasonality)")
    print(f"  series={len(t)}  train={split}  horizon={test_len}  "
          f"changepoints={model.n_changepoints}  periods={[p for p,_ in model.periods]}")
    print(f"  naive (last-value) RMSE: {n_rmse:6.3f}   MAE: {n_mae:6.3f}")
    print(f"  mean baseline      RMSE: {m_rmse:6.3f}")
    print(f"  Prophet forecast   RMSE: {p_rmse:6.3f}   MAE: {p_mae:6.3f}")
    print(f"  RMSE improvement over naive: {100 * (1 - p_rmse / n_rmse):.1f}%")
    print(f"  MAE  improvement over naive: {100 * (1 - p_mae / n_mae):.1f}%")
    print(f"  result: {'Prophet BEATS baselines' if p_rmse < min(n_rmse, m_rmse) else 'baseline wins'}")
    print("  sample forecasts vs truth (held-out tail):")
    for j in range(0, test_len, max(1, test_len // 6)):
        print(f"    day={int(te_t[j]):3d}  pred={pred[j]:7.3f}   true={te_y[j]:7.3f}   naive={naive[j]:7.3f}")
