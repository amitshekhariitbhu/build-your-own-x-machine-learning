import numpy as np

# Sales Prediction Model via Holt-Winters triple exponential smoothing, from scratch.
# Monthly sales are driven by a base LEVEL, a rising growth TREND, and a repeating
# 12-month SEASONAL pattern (holiday peaks / summer lulls) plus noise. Holt-Winters
# tracks three smoothed states -- level, trend, seasonal -- updating each with an
# EWMA-style rule, and forecasts the held-out tail from those states. We tune the
# (alpha, beta, gamma) smoothing rates by a small grid search on one-step error,
# then compare forecast RMSE/MAE against naive last-value and seasonal-naive baselines.


def make_sales(T=132, period=12, seed=0):
    # Synthetic monthly sales (units) with planted, recoverable structure.
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    month = t % period
    level, growth = 200.0, 1.4                       # base + rising trend
    season = np.array([-8, -14, 4, 10, 16, 6,        # planted seasonal shape:
                       -10, -18, -4, 12, 34, 48], float)  # Q4 spike, mid-year dip
    noise = rng.normal(0, 5.0, size=T)
    y = level + growth * t + season[month] + noise
    return y


class HoltWinters:
    """Additive triple exponential smoothing: level + trend + seasonal states."""

    def __init__(self, period=12, alpha=0.3, beta=0.05, gamma=0.3):
        self.m = period
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def fit(self, y):
        m, a, b, g = self.m, self.alpha, self.beta, self.gamma
        y = np.asarray(y, float)
        # Initialize states from the first two full seasons.
        level = y[:m].mean()
        trend = (y[m:2 * m].mean() - y[:m].mean()) / m
        seasonal = (y[:m] - level).astype(float)          # length-m seasonal indices
        for t in range(m, len(y)):
            s_prev = seasonal[t % m]
            last_level = level
            level = a * (y[t] - s_prev) + (1 - a) * (level + trend)
            trend = b * (level - last_level) + (1 - b) * trend
            seasonal[t % m] = g * (y[t] - level) + (1 - g) * s_prev
        self.level_, self.trend_, self.seasonal_ = level, trend, seasonal.copy()
        return self

    def forecast(self, steps):
        # Extrapolate level + trend and recycle the learned seasonal indices.
        h = np.arange(1, steps + 1)
        seas = self.seasonal_[(np.arange(steps)) % self.m]
        return self.level_ + h * self.trend_ + seas


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def tune(train, period, val=12):
    # Grid-search smoothing rates by one-step error on the tail of the training set.
    grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    inner, target = train[:-val], train[-val:]
    best, best_e = None, np.inf
    for a in grid:
        for b in grid:
            for g in grid:
                hw = HoltWinters(period, a, b, g).fit(inner)
                e = rmse(hw.forecast(val), target)
                if e < best_e:
                    best_e, best = e, (a, b, g)
    return best


if __name__ == "__main__":
    np.random.seed(0)
    period, horizon = 12, 12
    y = make_sales(T=132, period=period)
    train, test = y[:-horizon], y[-horizon:]          # hold out the final year

    a, b, g = tune(train, period)
    hw = HoltWinters(period, a, b, g).fit(train)
    pred = hw.forecast(horizon)

    # Baselines: repeat the last observed month; repeat the same month a year ago.
    naive = np.full(horizon, train[-1])
    seasonal_naive = train[-period:][:horizon]

    print("Sales Prediction Model -- Holt-Winters triple exponential smoothing")
    print(f"train months={len(train)}  test months={horizon}  "
          f"tuned alpha={a} beta={b} gamma={g}")
    print(f"{'model':<22}{'RMSE':>10}{'MAE':>10}")
    print(f"{'Holt-Winters':<22}{rmse(pred, test):>10.2f}{mae(pred, test):>10.2f}")
    print(f"{'naive (last value)':<22}{rmse(naive, test):>10.2f}"
          f"{mae(naive, test):>10.2f}")
    print(f"{'seasonal-naive':<22}{rmse(seasonal_naive, test):>10.2f}"
          f"{mae(seasonal_naive, test):>10.2f}")

    hw_r, naive_r = rmse(pred, test), rmse(naive, test)
    lift = 100.0 * (naive_r - hw_r) / naive_r
    print(f"\nRMSE improvement over naive: {lift:.1f}%")
    ok = hw_r < naive_r and hw_r < rmse(seasonal_naive, test)
    print("RESULT:", "PASS -- beats both baselines" if ok else "FAIL")
