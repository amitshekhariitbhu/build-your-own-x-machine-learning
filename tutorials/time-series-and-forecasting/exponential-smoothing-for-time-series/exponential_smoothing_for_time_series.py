import numpy as np

# Exponential Smoothing for Time Series (from scratch)
# ----------------------------------------------------
# We implement Holt-Winters style smoothing manually:
#   - Simple Exponential Smoothing (SES):   level only
#   - Holt (double):                        level + trend
#   - Holt-Winters (triple, additive):      level + trend + seasonality
# Each recursively blends the newest observation with prior state via
# smoothing weights (alpha, beta, gamma) in [0, 1]. No libraries do the work.


def simple_exponential_smoothing(y, alpha):
    """SES: level_t = alpha*y_t + (1-alpha)*level_{t-1}. Forecast = last level."""
    level = y[0]
    for t in range(1, len(y)):
        level = alpha * y[t] + (1.0 - alpha) * level
    return level  # flat forecast for any future horizon


class HoltWinters:
    """Triple exponential smoothing (additive trend + additive seasonality)."""

    def __init__(self, season_len, alpha=0.5, beta=0.1, gamma=0.3):
        self.m = season_len
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def fit(self, y):
        m = self.m
        # --- initialize state from the first two seasons ---
        level = np.mean(y[:m])
        trend = (np.mean(y[m:2 * m]) - np.mean(y[:m])) / m
        season = y[:m] - level  # additive seasonal indices

        for t in range(len(y)):
            s_idx = t % m
            prev_level = level
            deseason = y[t] - season[s_idx]
            level = self.alpha * deseason + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
            season[s_idx] = self.gamma * (y[t] - level) + (1 - self.gamma) * season[s_idx]

        self.level_, self.trend_, self.season_ = level, trend, season
        self.n_ = len(y)
        return self

    def forecast(self, h):
        """Forecast h steps ahead: level + k*trend + seasonal index."""
        out = np.empty(h)
        for k in range(1, h + 1):
            s_idx = (self.n_ + k - 1) % self.m
            out[k - 1] = self.level_ + k * self.trend_ + self.season_[s_idx]
        return out


def rmse(a, b):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def mae(a, b):
    return np.mean(np.abs(np.asarray(a) - np.asarray(b)))


if __name__ == "__main__":
    np.random.seed(0)

    # --- synthetic series: linear trend + additive seasonality + noise ---
    m = 12                               # season length (e.g. monthly)
    n = 156                              # 13 seasons
    t = np.arange(n)
    trend = 0.30 * t + 5.0
    season = 8.0 * np.sin(2 * np.pi * t / m)          # planted periodic signal
    noise = np.random.normal(0, 1.0, size=n)
    y = trend + season + noise

    # --- held-out tail split ---
    h = m                                # forecast one full season
    train, test = y[:-h], y[-h:]

    # --- Holt-Winters (triple exponential smoothing) ---
    hw = HoltWinters(season_len=m, alpha=0.4, beta=0.05, gamma=0.4).fit(train)
    hw_pred = hw.forecast(h)

    # --- baselines ---
    naive = np.full(h, train[-1])        # last-value carry-forward
    mean_base = np.full(h, train.mean()) # train mean
    ses_pred = np.full(h, simple_exponential_smoothing(train, alpha=0.4))

    print("Exponential Smoothing forecast vs baselines (held-out %d steps)" % h)
    print("-" * 60)
    print("Naive (last value)   RMSE=%6.3f  MAE=%6.3f" % (rmse(test, naive), mae(test, naive)))
    print("Mean baseline        RMSE=%6.3f  MAE=%6.3f" % (rmse(test, mean_base), mae(test, mean_base)))
    print("SES (level only)     RMSE=%6.3f  MAE=%6.3f" % (rmse(test, ses_pred), mae(test, ses_pred)))
    print("Holt-Winters (triple)RMSE=%6.3f  MAE=%6.3f" % (rmse(test, hw_pred), mae(test, hw_pred)))
    print("-" * 60)

    best_baseline = min(rmse(test, naive), rmse(test, mean_base), rmse(test, ses_pred))
    hw_rmse = rmse(test, hw_pred)
    improve = 100.0 * (best_baseline - hw_rmse) / best_baseline
    print("Best baseline RMSE=%6.3f -> Holt-Winters RMSE=%6.3f (%.1f%% lower)"
          % (best_baseline, hw_rmse, improve))
    print("RESULT:", "PASS - beats baseline" if hw_rmse < best_baseline else "FAIL")
