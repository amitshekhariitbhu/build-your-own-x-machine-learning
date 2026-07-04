import numpy as np

class DemandForecaster:
    """Product-demand predictor: ridge regression (from-scratch normal equations) on
    autoregressive lags of past demand plus KNOWN drivers -- weekly seasonality,
    promotion flag, and price. Exogenous drivers are available at forecast time, so
    the model can react to a planned promo/price the way a demand planner needs."""
    def __init__(self, n_lags=7, period=7, ridge=1.0):
        self.n_lags = n_lags       # how many past days of demand feed each prediction
        self.period = period       # weekly cycle length
        self.ridge = ridge         # L2 penalty stabilizes the least-squares solve
        self.w = None

    def _exog(self, t, promo, price):
        # Deterministic/known drivers at absolute time index t (all known ahead of time).
        t = np.atleast_1d(t).astype(float)
        feats = [np.ones_like(t),                                   # bias / base level
                 t / self.period,                                  # slow trend
                 np.sin(2 * np.pi * t / self.period),              # weekly seasonality
                 np.cos(2 * np.pi * t / self.period),
                 np.asarray(promo, float),                         # promotion running?
                 np.asarray(price, float)]                         # unit price (elasticity)
        return np.stack(feats, axis=1)

    def _design(self, series, promo, price, times):
        # One row per target time: [demand lag_1..lag_p, exogenous drivers at t].
        lags = np.stack([series[times - k] for k in range(1, self.n_lags + 1)], axis=1)
        return np.hstack([lags, self._exog(times, promo[times], price[times])])

    def fit(self, series, promo, price):
        series = np.asarray(series, float)
        self._trainlen = len(series)
        times = np.arange(self.n_lags, len(series))
        X = self._design(series, promo, price, times)
        y = series[times]
        # Ridge normal equations: w = (X'X + lambda I)^-1 X'y  (leave bias col unpenalized).
        A = X.T @ X
        reg = self.ridge * np.eye(X.shape[1]); reg[self.n_lags, self.n_lags] = 0.0
        self.w = np.linalg.solve(A + reg, X.T @ y)
        return self

    def predict_onestep(self, series, promo, price, times):
        # One-step-ahead demand using TRUE past demand + known drivers at each target time.
        X = self._design(np.asarray(series, float), promo, price, np.asarray(times))
        return np.clip(X @ self.w, 0.0, None)      # demand can't go negative

def make_demand(n=420, period=7, seed=0):
    # Synthetic daily product demand with recoverable latent structure:
    #   growth trend + weekly seasonality + price elasticity + promo lifts + AR(1) noise.
    np.random.seed(seed)
    t = np.arange(n)
    price = 10.0 + 2.0 * np.sin(2 * np.pi * t / 90) + np.random.randn(n) * 0.3  # drifting price
    promo = (np.random.rand(n) < 0.15).astype(float)                            # ~15% promo days
    base = 50.0 + 0.05 * t                                    # baseline + slow growth
    seasonal = 12.0 * np.sin(2 * np.pi * t / period)          # weekly demand cycle
    elasticity = -3.0 * (price - price.mean())               # cheaper -> more demand
    lift = 25.0 * promo                                       # promotions spike demand
    noise = np.zeros(n)
    for i in range(1, n):                                     # AR(1) noise -> lags informative
        noise[i] = 0.6 * noise[i - 1] + np.random.randn() * 2.0
    demand = np.clip(base + seasonal + elasticity + lift + noise, 0.0, None)
    return demand, promo, price

if __name__ == "__main__":
    np.random.seed(0)
    period, test_len = 7, 60
    demand, promo, price = make_demand(n=420, period=period)
    split = len(demand) - test_len                # forecast the held-out tail of days

    model = DemandForecaster(n_lags=7, period=period, ridge=1.0)
    model.fit(demand[:split], promo[:split], price[:split])

    idx = np.arange(split, len(demand))           # held-out day indices
    pred = model.predict_onestep(demand, promo, price, idx)
    true = demand[idx]
    naive = demand[idx - 1]                        # baseline: yesterday's demand
    seasonal_naive = demand[idx - period]          # baseline: same weekday last week
    mean_base = np.full_like(true, demand[:split].mean())  # baseline: train mean

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, m_mae = rmse(pred, true), mae(pred, true)
    n_rmse, n_mae = rmse(naive, true), mae(naive, true)
    s_rmse = rmse(seasonal_naive, true)
    mn_rmse = rmse(mean_base, true)
    best_base = min(n_rmse, s_rmse, mn_rmse)

    print("Product demand prediction (ridge AR + promo/price/seasonal drivers)")
    print(f"  days={len(demand)}  lags={model.n_lags}  weekly_period={period}  test_tail={test_len}")
    print(f"  baseline mean          RMSE: {mn_rmse:6.3f}")
    print(f"  baseline last-value    RMSE: {n_rmse:6.3f}   MAE: {n_mae:6.3f}")
    print(f"  baseline seasonal-naive RMSE: {s_rmse:6.3f}")
    print(f"  demand model           RMSE: {m_rmse:6.3f}   MAE: {m_mae:6.3f}")
    print(f"  RMSE improvement over best baseline: {100*(1-m_rmse/best_base):.1f}%")
    print(f"  MAE  improvement over last-value:    {100*(1-m_mae/n_mae):.1f}%")
    print(f"  result: {'demand model BEATS all baselines' if m_rmse < best_base else 'baseline wins'}")
    print("  sample forecasts vs truth (held-out days, promo flag shown):")
    for j in range(0, len(true), max(1, len(true) // 6)):
        tag = "PROMO" if promo[idx][j] else "     "
        print(f"    pred={pred[j]:6.2f}   true={true[j]:6.2f}   yesterday={naive[j]:6.2f}  {tag}")
