import numpy as np

class LoadForecaster:
    """Electricity demand / load forecaster built from scratch with ridge regression
    (normal equations) on autoregressive lags plus KNOWN calendar & weather drivers:
      - recent load lags (short memory + same-hour-yesterday),
      - Fourier harmonics for the daily (24h) and weekly (168h) cycles,
      - temperature split into heating & cooling degrees -> the U-shaped load curve,
      - a weekend flag.
    Weather and calendar are known at forecast time, so a planner can react to a
    forecast heatwave or a holiday the way a real load-forecasting desk does."""
    def __init__(self, lags=(1, 2, 24, 25, 168), day=24, week=168, ridge=5.0, comfort=20.0):
        self.lags = tuple(lags)      # which past-load offsets feed each prediction
        self.day = day               # hours per daily cycle
        self.week = week             # hours per weekly cycle
        self.ridge = ridge           # L2 penalty stabilizes the least-squares solve
        self.comfort = comfort       # thermal-neutral temperature (deg C)
        self.w = None

    def _exog(self, t, temp):
        # Deterministic/known drivers at absolute hour index t (all known ahead of time).
        t = np.atleast_1d(t).astype(float); temp = np.asarray(temp, float)
        cool = np.maximum(temp - self.comfort, 0.0)   # air-conditioning load
        heat = np.maximum(self.comfort - temp, 0.0)   # heating load
        weekend = ((t // self.day).astype(int) % 7 >= 5).astype(float)
        feats = [np.ones_like(t),                          # bias / base level
                 np.sin(2*np.pi*t/self.day),  np.cos(2*np.pi*t/self.day),   # daily cycle
                 np.sin(4*np.pi*t/self.day),  np.cos(4*np.pi*t/self.day),   # daily 2nd harmonic
                 np.sin(2*np.pi*t/self.week), np.cos(2*np.pi*t/self.week),  # weekly cycle
                 cool, heat, weekend]
        return np.stack(feats, axis=1)

    def _design(self, series, temp, times):
        # One row per target hour: [load lags..., known exogenous drivers at t].
        lags = np.stack([series[times - k] for k in self.lags], axis=1)
        return np.hstack([lags, self._exog(times, temp[times])])

    def fit(self, series, temp):
        series = np.asarray(series, float)
        times = np.arange(max(self.lags), len(series))
        X = self._design(series, temp, times); y = series[times]
        # Ridge normal equations: w = (X'X + lambda I)^-1 X'y (leave bias col unpenalized).
        A = X.T @ X
        reg = self.ridge * np.eye(X.shape[1]); reg[len(self.lags), len(self.lags)] = 0.0
        self.w = np.linalg.solve(A + reg, X.T @ y)
        return self

    def predict_onestep(self, series, temp, times):
        # One-step-ahead load using TRUE recent load + known drivers at each target hour.
        X = self._design(np.asarray(series, float), temp, np.asarray(times))
        return np.clip(X @ self.w, 0.0, None)          # load can't go negative

def make_load(n=1440, day=24, seed=0):
    # Synthetic hourly grid load with recoverable latent structure:
    #   base + growth + daily/weekly shape + weekend dip + temperature U-curve + AR(1) noise.
    np.random.seed(seed)
    t = np.arange(n); hour = t % day; dow = (t // day) % 7
    # Temperature: seasonal drift + daily swing (afternoon warm, night cool) + weather noise.
    temp = 20.0 + 8.0*np.sin(2*np.pi*t/(day*30)) + 6.0*np.sin(2*np.pi*(hour-9)/day) \
        + np.random.randn(n)*1.2
    daily = 20.0*np.sin(2*np.pi*(hour-15)/day) + 8.0*np.sin(4*np.pi*hour/day)  # midday+evening peaks
    weekend = -12.0 * (dow >= 5)                                               # quieter weekends
    cool = 2.6*np.maximum(temp - 20.0, 0.0); heat = 1.8*np.maximum(20.0 - temp, 0.0)
    base = 100.0 + 0.01*t                                                      # base demand + slow growth
    noise = np.zeros(n)
    for i in range(1, n):                                                      # AR(1) -> lags informative
        noise[i] = 0.7*noise[i-1] + np.random.randn()*2.5
    load = np.clip(base + daily + weekend + cool + heat + noise, 0.0, None)
    return load, temp

if __name__ == "__main__":
    np.random.seed(0)
    day, week, test_len = 24, 168, 168                 # forecast the held-out final week
    load, temp = make_load(n=1440, day=day)
    split = len(load) - test_len

    model = LoadForecaster(lags=(1, 2, 24, 25, 168), day=day, week=week, ridge=5.0)
    model.fit(load[:split], temp[:split])

    idx = np.arange(split, len(load))                  # held-out hour indices
    pred = model.predict_onestep(load, temp, idx)
    true = load[idx]
    naive = load[idx - 1]                              # baseline: previous hour
    daily_naive = load[idx - day]                      # baseline: same hour yesterday
    week_naive = load[idx - week]                      # baseline: same hour last week
    mean_base = np.full_like(true, load[:split].mean())  # baseline: train mean

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b)**2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, m_mae = rmse(pred, true), mae(pred, true)
    bases = {"mean": mean_base, "last-hour": naive, "same-hour-yesterday": daily_naive,
             "same-hour-last-week": week_naive}
    base_rmse = {k: rmse(v, true) for k, v in bases.items()}
    best_name = min(base_rmse, key=base_rmse.get); best = base_rmse[best_name]

    print("Demand & load forecasting (ridge AR + daily/weekly Fourier + temperature U-curve)")
    print(f"  hours={len(load)}  lags={model.lags}  daily={day}  weekly={week}  test_tail={test_len}")
    for k, v in base_rmse.items():
        print(f"  baseline {k:<20s} RMSE: {v:7.3f}")
    print(f"  load model               RMSE: {m_rmse:7.3f}   MAE: {m_mae:7.3f}")
    print(f"  RMSE improvement over best baseline ({best_name}): {100*(1-m_rmse/best):.1f}%")
    print(f"  result: {'load model BEATS all baselines' if m_rmse < best else 'baseline wins'}")
    print("  sample forecasts vs truth (held-out hours, temp shown):")
    for j in range(0, len(true), max(1, len(true) // 6)):
        tag = "WKND" if ((idx[j] // day) % 7) >= 5 else "    "
        print(f"    hour={idx[j]:4d}  pred={pred[j]:7.2f}  true={true[j]:7.2f}  temp={temp[idx][j]:5.1f}C  {tag}")
