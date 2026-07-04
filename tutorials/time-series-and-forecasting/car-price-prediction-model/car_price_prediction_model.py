import numpy as np

# Car Price Prediction Model (from scratch)
# -----------------------------------------
# We forecast a monthly USED-CAR market price index. Such a series carries a
# slow depreciation TREND, an annual SEASONAL swing (spring buying season lifts
# prices, winter softens them), and mean-reverting AR deviations. We fit ONE
# linear-regression-on-lags model whose design matrix packs, all by hand:
#   - intercept + linear time              -> the depreciation trend
#   - sin/cos harmonics of the year        -> smooth annual seasonality
#   - p autoregressive lags y[t-1..t-p]    -> short-term momentum / reversion
# Coefficients come from ridge-regularized normal equations (np.linalg.solve).
# One-step-ahead forecasts on a HELD-OUT tail use only true past values, so
# beating a naive last-value baseline proves the planted structure was learned.


class CarPricePredictor:
    """Ridge linear regression on time + seasonal harmonics + AR lags."""

    def __init__(self, period=12, n_harmonics=2, ar_order=3, ridge=1e-3):
        self.period = period            # seasonal cycle length (12 = annual)
        self.n_harmonics = n_harmonics  # sin/cos pairs for seasonality
        self.ar_order = ar_order        # number of autoregressive lags
        self.ridge = ridge              # L2 stabilizer for normal equations

    def _features(self, y, t):
        # Build one design row set for absolute times t using history y.
        # y[t-1..t-p] must exist, so callers pass t >= ar_order.
        t = np.asarray(t)
        cols = [np.ones(len(t)), t.astype(float)]          # trend
        for h in range(1, self.n_harmonics + 1):           # seasonal harmonics
            ang = 2 * np.pi * h * t / self.period
            cols.append(np.sin(ang))
            cols.append(np.cos(ang))
        for j in range(1, self.ar_order + 1):              # AR lags
            cols.append(y[t - j])
        return np.stack(cols, axis=1)

    def fit(self, prices):
        y = np.asarray(prices, float)
        n = len(y)
        t = np.arange(self.ar_order, n)                    # rows with full lags
        X = self._features(y, t)
        target = y[t]
        G = X.T @ X + self.ridge * np.eye(X.shape[1])      # ridge normal equations
        self.coef = np.linalg.solve(G, X.T @ target)
        self.n_train = n
        return self

    def predict(self, prices, ks):
        # One-step-ahead price for target times ks using true past values only.
        y = np.asarray(prices, float)
        X = self._features(y, np.asarray(ks))
        return X @ self.coef


def make_car_price_series(n=180, seed=0):
    # Monthly used-car price index: gentle depreciation trend + annual season +
    # AR(2) mean-reverting residual + small noise. Structured and recoverable,
    # yet a last-value baseline always lags the trend and seasonal turns.
    np.random.seed(seed)
    t = np.arange(n)
    trend = 26000.0 - 22.0 * t + 0.06 * t ** 2             # depreciate then flatten
    season = 900.0 * np.sin(2 * np.pi * t / 12.0) \
        + 350.0 * np.cos(2 * np.pi * t / 6.0)              # spring peak, subcycle
    resid = np.zeros(n)
    for i in range(2, n):                                  # AR(2) deviations
        resid[i] = 0.55 * resid[i - 1] - 0.20 * resid[i - 2] \
            + 90.0 * np.random.randn()
    return trend + season + resid


if __name__ == "__main__":
    np.random.seed(0)
    prices = make_car_price_series(n=180)
    test_len = 36
    split = len(prices) - test_len                 # forecast the held-out tail

    model = CarPricePredictor(period=12, n_harmonics=2, ar_order=3)
    model.fit(prices[:split])

    ks = np.arange(split, len(prices))             # one-step targets in the tail
    pred = model.predict(prices, ks)
    true = prices[ks]
    naive = prices[ks - 1]                          # baseline: last observed value
    mean_base = np.full_like(true, prices[:split].mean())   # train-mean baseline

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, n_rmse, mean_rmse = rmse(pred, true), rmse(naive, true), rmse(mean_base, true)
    m_mae, n_mae = mae(pred, true), mae(naive, true)

    # Directional accuracy: did we call the up/down move right? Random = 50%.
    dir_acc = float(np.mean((pred > naive) == (true > naive)))

    print("Car price prediction (trend + seasonal harmonics + AR lags, ridge LS)")
    print(f"  months={len(prices)}  train={split}  test_tail={test_len}  ar_order={model.ar_order}")
    print(f"  learned coeffs: intercept={model.coef[0]:.1f}  time_slope={model.coef[1]:.2f}")
    print(f"  baseline train-mean  RMSE: {mean_rmse:9.3f}")
    print(f"  baseline last-value  RMSE: {n_rmse:9.3f}   MAE: {n_mae:9.3f}")
    print(f"  model  one-step      RMSE: {m_rmse:9.3f}   MAE: {m_mae:9.3f}")
    print(f"  RMSE improvement over last-value: {100 * (1 - m_rmse / n_rmse):.1f}%")
    print(f"  MAE  improvement over last-value: {100 * (1 - m_mae / n_mae):.1f}%")
    print(f"  directional accuracy: {dir_acc:.3f}   (random baseline 0.500)")
    beat = m_rmse < n_rmse and m_rmse < mean_rmse and dir_acc > 0.5
    print(f"  result: {'model BEATS last-value + mean baselines' if beat else 'baseline wins'}")
    print("  sample one-step forecasts vs truth (held-out tail):")
    for j in range(0, len(true), max(1, len(true) // 6)):
        print(f"    pred={pred[j]:10.2f}   true={true[j]:10.2f}   naive={naive[j]:10.2f}")
