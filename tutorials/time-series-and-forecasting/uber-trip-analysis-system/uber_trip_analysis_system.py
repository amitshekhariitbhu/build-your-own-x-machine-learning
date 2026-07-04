import numpy as np

# Uber Trip Analysis System (from scratch)
# ----------------------------------------
# Task: forecast hourly Uber ride demand. Real ride demand is driven by
# strong DAILY rush-hour cycles and a WEEKLY (weekday vs weekend) rhythm,
# riding on a slow growth trend. We model this with a seasonal
# AutoRegressor: an ordinary-least-squares linear model on lagged demand
#   lag 1   (last hour)        -> local momentum
#   lag 24  (same hour, 1 day) -> daily rush-hour cycle
#   lag 168 (same hour, 1 week)-> weekly weekday/weekend cycle
# plus sin/cos hour-of-day features. Coefficients are solved by hand via
# the normal equations (np.linalg). Forecasting is recursive: each new
# prediction is fed back in as history for the next step. No ML library.


class SeasonalAutoRegressor:
    """OLS linear regression on seasonal lags + hour-of-day sinusoids."""

    def __init__(self, lags=(1, 24, 168), season=24):
        self.lags = tuple(lags)
        self.season = season

    def _row(self, hist, hour):
        """Build one feature row from history (hist[-1] is the latest hour)."""
        row = [1.0]                                   # bias
        for L in self.lags:
            row.append(hist[-L])                      # demand L hours ago
        row.append(np.sin(2 * np.pi * hour / self.season))
        row.append(np.cos(2 * np.pi * hour / self.season))
        return row

    def fit(self, y):
        maxlag = max(self.lags)
        X, Y = [], []
        for t in range(maxlag, len(y)):
            X.append(self._row(y[:t], t % self.season))
            Y.append(y[t])
        X, Y = np.asarray(X), np.asarray(Y)
        # normal equations: beta = (X'X)^-1 X'y  (solved via lstsq for stability)
        self.beta_, *_ = np.linalg.lstsq(X, Y, rcond=None)
        self.n_ = len(y)
        return self

    def forecast(self, y_train, h):
        """Recursive multi-step forecast: predictions become future history."""
        hist = list(y_train)
        out = np.empty(h)
        for k in range(h):
            hour = (self.n_ + k) % self.season
            row = np.asarray(self._row(hist, hour))
            pred = float(row @ self.beta_)
            out[k] = pred
            hist.append(pred)                         # feed prediction forward
        return out


def make_uber_demand(weeks=6, seed=0):
    """Synthetic hourly ride counts: growth trend + daily + weekly cycles."""
    np.random.seed(seed)
    n = weeks * 7 * 24
    t = np.arange(n)
    hour = t % 24
    dow = (t // 24) % 7                                # day of week (0=Mon)

    trend = 40.0 + 0.010 * t                           # slow demand growth
    # daily rush-hour shape: morning (8h) and evening (18h) peaks
    daily = 22 * np.exp(-((hour - 8) ** 2) / 6.0) + 30 * np.exp(-((hour - 18) ** 2) / 6.0)
    weekend = np.where(dow >= 5, 18.0, 0.0)            # weekend nightlife bump
    late_night = np.where((dow >= 5) & (hour < 4), 25.0, 0.0)
    noise = np.random.normal(0, 3.0, size=n)
    y = trend + daily + weekend + late_night + noise
    return np.clip(y, 0, None)                          # ride counts are >= 0


def rmse(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def mae(a, b):
    return np.mean(np.abs(np.asarray(a) - np.asarray(b)))


if __name__ == "__main__":
    y = make_uber_demand(weeks=6, seed=0)

    h = 48                                             # forecast next 2 days
    train, test = y[:-h], y[-h:]

    model = SeasonalAutoRegressor(lags=(1, 24, 168), season=24).fit(train)
    pred = model.forecast(train, h)

    # --- baselines (no leakage from the test tail) ---
    naive = np.full(h, train[-1])                       # carry last hour forward
    mean_base = np.full(h, train.mean())                # overall average demand
    seasonal_naive = np.tile(train[-24:], 2)[:h]        # repeat yesterday's day

    print("Uber Trip Analysis: hourly demand forecast (held-out %d hours)" % h)
    print("-" * 62)
    print("Naive (last value)      RMSE=%6.3f  MAE=%6.3f" % (rmse(test, naive), mae(test, naive)))
    print("Mean baseline           RMSE=%6.3f  MAE=%6.3f" % (rmse(test, mean_base), mae(test, mean_base)))
    print("Seasonal naive (24h)    RMSE=%6.3f  MAE=%6.3f" % (rmse(test, seasonal_naive), mae(test, seasonal_naive)))
    print("Seasonal AutoRegressor  RMSE=%6.3f  MAE=%6.3f" % (rmse(test, pred), mae(test, pred)))
    print("-" * 62)

    best_base = min(rmse(test, naive), rmse(test, mean_base), rmse(test, seasonal_naive))
    ar_rmse = rmse(test, pred)
    improve = 100.0 * (best_base - ar_rmse) / best_base
    print("Best baseline RMSE=%6.3f -> AutoRegressor RMSE=%6.3f (%.1f%% lower)"
          % (best_base, ar_rmse, improve))
    print("RESULT:", "PASS - beats baseline" if ar_rmse < best_base else "FAIL")
