import numpy as np

# Calorie Burn Prediction Model via autoregressive linear regression on lags, from scratch.
# Daily workout calories are driven by a slow fitness TREND (longer/harder sessions over
# time), a repeating 7-day WEEKLY pattern (long weekend run, Sunday rest), and an AR(1)
# residual that gives short-term day-to-day MOMENTUM. We fit one linear model whose features
# are lagged values (lag-1, lag-7), a time index, and day-of-week dummies, solving the normal
# equations with least squares. The held-out tail is forecast RECURSIVELY (feeding predictions
# back as lags), then compared against naive last-value, seasonal-naive, and mean baselines.


def make_calories(T=210, period=7, seed=0):
    # Synthetic daily calories burned with planted, recoverable structure.
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    dow = t % period
    base, growth = 320.0, 0.6                              # base + rising fitness trend
    weekly = np.array([300, 320, 360, 320, 380, 540, 190],  # Mon..Sun: weekend peak, Sun rest
                      float)
    season = weekly[dow] - weekly.mean()                  # center so base stays meaningful
    e = np.zeros(T)                                        # AR(1) momentum in the residual
    for i in range(1, T):
        e[i] = 0.5 * e[i - 1] + rng.normal(0, 14)
    return base + growth * t + season + e


class ARSeasonalRegressor:
    """Linear regression on lagged values + time trend + day-of-week dummies."""

    def __init__(self, lags=(1, 7), period=7):
        self.lags, self.period = lags, period

    def _row(self, series, t):
        # Feature row for the target at absolute time index t (uses history series[:t]).
        feats = [1.0]                                      # intercept
        for lag in self.lags:
            feats.append(series[t - lag])                 # autoregressive lags
        feats.append(t / 100.0)                           # scaled linear trend
        dow = t % self.period                             # day-of-week one-hot (drop last)
        onehot = [0.0] * (self.period - 1)
        if dow < self.period - 1:
            onehot[dow] = 1.0
        return feats + onehot

    def fit(self, y):
        y = np.asarray(y, float)
        start = max(self.lags)
        X = np.array([self._row(y, t) for t in range(start, len(y))])
        z = y[start:]
        self.coef_, *_ = np.linalg.lstsq(X, z, rcond=None)  # least-squares normal equations
        return self

    def forecast(self, y_train, steps):
        # Recursive multi-step forecast: predictions are fed back in as future lags.
        series = [float(v) for v in y_train]
        n = len(series)
        preds = []
        for i in range(steps):
            t = n + i
            p = float(np.dot(self.coef_, self._row(np.asarray(series), t)))
            preds.append(p)
            series.append(p)
        return np.array(preds)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


if __name__ == "__main__":
    np.random.seed(0)
    period, horizon = 7, 21
    y = make_calories(T=210, period=period)
    train, test = y[:-horizon], y[-horizon:]              # hold out the final 3 weeks

    model = ARSeasonalRegressor(lags=(1, period), period=period).fit(train)
    pred = model.forecast(train, horizon)

    # Baselines: repeat last day; repeat same weekday a week ago; repeat the mean.
    naive = np.full(horizon, train[-1])
    seasonal_naive = np.array([train[-period + (i % period)] for i in range(horizon)])
    mean_base = np.full(horizon, train.mean())

    print("Calorie Burn Prediction Model -- autoregressive linear regression on lags")
    print(f"train days={len(train)}  test days={horizon}  lags=(1,{period})  period={period}")
    print(f"{'model':<24}{'RMSE':>10}{'MAE':>10}")
    print(f"{'AR + seasonal reg':<24}{rmse(pred, test):>10.2f}{mae(pred, test):>10.2f}")
    print(f"{'naive (last value)':<24}{rmse(naive, test):>10.2f}{mae(naive, test):>10.2f}")
    print(f"{'seasonal-naive (t-7)':<24}{rmse(seasonal_naive, test):>10.2f}"
          f"{mae(seasonal_naive, test):>10.2f}")
    print(f"{'mean':<24}{rmse(mean_base, test):>10.2f}{mae(mean_base, test):>10.2f}")

    m_r = rmse(pred, test)
    best_base = min(rmse(naive, test), rmse(seasonal_naive, test), rmse(mean_base, test))
    lift = 100.0 * (best_base - m_r) / best_base
    print(f"\nRMSE improvement over best baseline: {lift:.1f}%")
    print("RESULT:", "PASS -- beats every baseline" if m_r < best_base else "FAIL")
