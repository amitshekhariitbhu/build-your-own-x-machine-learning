import numpy as np

# House Price Prediction Model (from scratch)
# -------------------------------------------
# A monthly house-price INDEX carries a rising (slightly accelerating) TREND,
# an annual SEASONAL swing (spring buying season), and mean-reverting noise.
# We forecast it with a linear regression on engineered features, all math by
# hand via the ridge normal equations w = (X'X + R)^-1 X'y:
#   - intercept + linear + quadratic TREND        (t, t^2, time normalized)
#   - Fourier SEASONAL terms  sin/cos(2*pi*h*t/12) for h harmonics
#   - AR LAGS  price[t-1..t-p]                     (autoregression on the series)
# A one-step forecast uses only actual PAST prices for the lags, so predicting
# the held-out tail is honest. Naive last-value / seasonal-naive baselines lag
# the trend and season, so beating their RMSE proves the model recovered the
# planted structure.


class HousePricePredictor:
    """Linear regression on trend + Fourier-seasonal + AR-lag features (ridge)."""

    def __init__(self, period=12, n_lags=3, n_harmonics=2, ridge=1e-2):
        self.period = period          # seasonal cycle length (12 = annual)
        self.n_lags = n_lags          # autoregressive lags of the price series
        self.n_harmonics = n_harmonics  # sin/cos pairs for the seasonal shape
        self.ridge = ridge            # L2 stabilizer for the normal equations

    def _design(self, y, idx):
        # Build the feature matrix for target times idx (each idx >= n_lags),
        # pulling lag values from the observed series y.
        idx = np.asarray(idx)
        tn = idx * self.tscale                      # normalized time (~0..1)
        cols = [np.ones(len(idx)), tn, tn ** 2]     # intercept + trend
        for h in range(1, self.n_harmonics + 1):    # seasonal Fourier terms
            ang = 2.0 * np.pi * h * idx / self.period
            cols.append(np.sin(ang))
            cols.append(np.cos(ang))
        for j in range(1, self.n_lags + 1):         # AR lags: price[t-j]
            cols.append(y[idx - j])
        return np.column_stack(cols)

    def fit(self, prices):
        y = np.asarray(prices, float)
        n = len(y)
        self.tscale = 1.0 / n                        # fix time scale from train span
        idx = np.arange(self.n_lags, n)             # first n_lags have no full lags
        X = self._design(y, idx)
        target = y[idx]
        d = X.shape[1]
        R = self.ridge * np.eye(d)
        R[0, 0] = 0.0                                # do not penalize the intercept
        self.w = np.linalg.solve(X.T @ X + R, X.T @ target)
        return self

    def predict(self, prices, idx):
        # One-step-ahead price for target times idx, using ONLY prices[idx-lag].
        X = self._design(np.asarray(prices, float), idx)
        return X @ self.w


def make_house_price_series(n=180, seed=0):
    # Monthly house-price index: accelerating upward trend + annual seasonality
    # (spring peak) + AR(1) mean-reverting noise. The structure is recoverable,
    # yet a last-value baseline always lags the trend and seasonal swing.
    np.random.seed(seed)
    t = np.arange(n)
    trend = 200.0 + 1.4 * t + 0.010 * t ** 2                     # rising prices
    season = 12.0 * np.sin(2 * np.pi * t / 12.0) \
        + 5.0 * np.cos(2 * np.pi * t / 6.0)                      # spring buying season
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = 0.55 * resid[i - 1] + 3.0 * np.random.randn()  # AR(1) deviations
    return trend + season + resid


if __name__ == "__main__":
    np.random.seed(0)
    prices = make_house_price_series(n=180)
    test_len = 36
    split = len(prices) - test_len               # forecast the held-out tail

    model = HousePricePredictor(period=12, n_lags=3, n_harmonics=2).fit(prices[:split])

    idx = np.arange(split, len(prices))          # one-step targets in the test tail
    pred = model.predict(prices, idx)
    true = prices[idx]
    naive = prices[idx - 1]                       # baseline: last observed value
    seas_naive = prices[idx - 12]                 # baseline: value one year ago
    mean_base = np.full_like(true, prices[:split].mean())        # train-mean baseline

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, n_rmse = rmse(pred, true), rmse(naive, true)
    s_rmse, mean_rmse = rmse(seas_naive, true), rmse(mean_base, true)
    m_mae, n_mae = mae(pred, true), mae(naive, true)

    # Directional accuracy: did we call the up/down move right? Random = 50%.
    dir_acc = float(np.mean((pred > naive) == (true > naive)))

    print("House price prediction (trend + Fourier seasonality + AR lags, ridge regression)")
    print(f"  months={len(prices)}  train={split}  test_tail={test_len}"
          f"  lags={model.n_lags}  harmonics={model.n_harmonics}")
    print(f"  baseline train-mean   RMSE: {mean_rmse:8.3f}")
    print(f"  baseline seasonal(-12)RMSE: {s_rmse:8.3f}")
    print(f"  baseline last-value   RMSE: {n_rmse:8.3f}   MAE: {n_mae:8.3f}")
    print(f"  model  one-step       RMSE: {m_rmse:8.3f}   MAE: {m_mae:8.3f}")
    print(f"  RMSE improvement over last-value: {100 * (1 - m_rmse / n_rmse):.1f}%")
    print(f"  MAE  improvement over last-value: {100 * (1 - m_mae / n_mae):.1f}%")
    print(f"  directional accuracy: {dir_acc:.3f}   (random baseline 0.500)")
    beat = m_rmse < n_rmse and m_rmse < s_rmse and m_rmse < mean_rmse and dir_acc > 0.5
    print(f"  result: {'model BEATS last-value + seasonal + mean baselines' if beat else 'baseline wins'}")
    print("  sample one-step forecasts vs truth (held-out tail):")
    for j in range(0, len(true), max(1, len(true) // 6)):
        print(f"    pred={pred[j]:9.2f}   true={true[j]:9.2f}   naive={naive[j]:9.2f}")
