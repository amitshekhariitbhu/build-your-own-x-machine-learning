import numpy as np

class OrderVolumeForecaster:
    """Daily order-volume forecaster built from scratch as a calendar-regression /
    seasonal-decomposition model. Order counts are driven by KNOWN calendar effects:
    a growth trend, a day-of-week profile (weekend spikes), and a slow annual cycle.
    We regress volume on deterministic calendar features via ridge normal equations,
    so the fitted level/trend/seasonal parts extrapolate to ANY future date -- exactly
    what a multi-week order-volume forecast needs. No AR lags required at forecast time."""

    def __init__(self, ridge=1e-3, t_scale=100.0, year=365.0):
        self.ridge = ridge        # tiny L2 for a stable, well-conditioned solve
        self.t_scale = t_scale    # scale time index so trend column is well-conditioned
        self.year = year          # length of the annual cycle in days
        self.w = None

    def _features(self, t):
        # Deterministic calendar design row for absolute day index t (known for any future t).
        t = np.atleast_1d(t).astype(float)
        dow = (t.astype(int) % 7)
        onehot = np.eye(7)[dow]                              # day-of-week seasonal dummies
        trend = (t / self.t_scale)[:, None]                 # linear growth term
        annual = np.stack([np.sin(2 * np.pi * t / self.year),   # slow yearly cycle
                           np.cos(2 * np.pi * t / self.year)], axis=1)
        return np.hstack([onehot, trend, annual])           # 7 + 1 + 2 = 10 columns

    def fit(self, volume, start=0):
        # Fit on absolute day indices start..start+len-1 so future dates line up correctly.
        volume = np.asarray(volume, float)
        t = np.arange(start, start + len(volume))
        X = self._features(t)
        A = X.T @ X + self.ridge * np.eye(X.shape[1])       # ridge-regularized normal equations
        self.w = np.linalg.solve(A, X.T @ volume)
        return self

    def predict(self, t):
        # Forecast order volume at absolute day indices t (deterministic multi-step).
        return np.clip(self._features(t) @ self.w, 0.0, None)   # volume can't go negative


def make_orders(n=364, seed=0):
    # Synthetic daily order volume with recoverable latent structure:
    #   growth trend + fixed weekday profile (weekend spike) + annual cycle + AR(1) noise.
    np.random.seed(seed)
    t = np.arange(n)
    base = 200.0 + 0.55 * t                                  # baseline + steady growth
    weekday = np.array([-35, -25, -10, 5, 20, 70, 55])       # Mon..Sun order pattern
    weekly = weekday[t % 7]                                  # weekly seasonality
    annual = 40.0 * np.sin(2 * np.pi * t / 365.0)            # slow yearly demand swing
    noise = np.zeros(n)
    for i in range(1, n):                                    # AR(1) noise -> realistic wiggle
        noise[i] = 0.5 * noise[i - 1] + np.random.randn() * 6.0
    orders = np.clip(base + weekly + annual + noise, 0.0, None)
    return np.round(orders)                                  # order counts are integers


if __name__ == "__main__":
    np.random.seed(0)
    test_len = 28                                            # forecast the last 4 weeks
    orders = make_orders(n=364)
    split = len(orders) - test_len

    model = OrderVolumeForecaster().fit(orders[:split], start=0)

    idx = np.arange(split, len(orders))                     # held-out day indices
    pred = model.predict(idx)
    true = orders[idx]
    last_value = np.full_like(true, orders[split - 1])       # baseline: repeat last observed day
    mean_base = np.full_like(true, orders[:split].mean())    # baseline: train mean
    seasonal_naive = orders[idx - 7]                         # baseline: same weekday last week

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, m_mae = rmse(pred, true), mae(pred, true)
    lv_rmse = rmse(last_value, true)
    mn_rmse = rmse(mean_base, true)
    sn_rmse, sn_mae = rmse(seasonal_naive, true), mae(seasonal_naive, true)
    best_base = min(lv_rmse, mn_rmse, sn_rmse)

    print("Order volume prediction (calendar-regression: trend + weekday + annual)")
    print(f"  days={len(orders)}  features=10  test_tail={test_len} (multi-step forecast)")
    print(f"  baseline last-value      RMSE: {lv_rmse:7.3f}")
    print(f"  baseline train-mean      RMSE: {mn_rmse:7.3f}")
    print(f"  baseline seasonal-naive  RMSE: {sn_rmse:7.3f}   MAE: {sn_mae:6.3f}")
    print(f"  order-volume model       RMSE: {m_rmse:7.3f}   MAE: {m_mae:6.3f}")
    print(f"  RMSE improvement over best baseline: {100 * (1 - m_rmse / best_base):.1f}%")
    print(f"  result: {'MODEL BEATS all baselines' if m_rmse < best_base else 'baseline wins'}")
    print("  sample forecasts vs truth (held-out days):")
    for j in range(0, len(true), max(1, len(true) // 7)):
        print(f"    day={idx[j]:3d}  pred={pred[j]:7.2f}   true={true[j]:7.2f}"
              f"   last-week={seasonal_naive[j]:7.2f}")
