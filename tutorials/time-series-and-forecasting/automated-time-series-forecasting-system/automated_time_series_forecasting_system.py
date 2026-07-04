import numpy as np

# Automated forecasting = a pool of from-scratch models, each self-tuning its own
# hyperparameters, plus a selector that picks the winner by validation error.

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def detect_period(y, lo=4, hi=40):
    # Dominant season = lag with the highest autocorrelation (auto seasonality guess).
    y = y - y.mean()
    hi = min(hi, len(y) // 2)
    ac = [np.dot(y[:-k], y[k:]) / np.dot(y, y) for k in range(lo, hi)]
    return lo + int(np.argmax(ac))

class NaiveLast:
    """Baseline: forecast = last observed value, held flat."""
    name = "Naive(last)"
    def fit(self, y):
        self.last = y[-1]; return self
    def forecast(self, h):
        return np.full(h, self.last)

class MovingAverage:
    """Flat forecast at the mean of the last k points."""
    def __init__(self, k=8): self.k = k; self.name = f"MA(k={k})"
    def fit(self, y):
        self.val = y[-self.k:].mean(); return self
    def forecast(self, h):
        return np.full(h, self.val)

class SES:
    """Simple exponential smoothing; alpha tuned by one-step SSE."""
    name = "SES"
    def fit(self, y):
        best = (np.inf, 0.3)
        for a in np.linspace(0.05, 0.95, 19):
            l, sse = y[0], 0.0
            for t in range(1, len(y)):
                sse += (y[t] - l) ** 2                 # l is the one-step forecast
                l = a * y[t] + (1 - a) * l
            if sse < best[0]: best = (sse, a)
        self.a = best[1]
        l = y[0]
        for t in range(1, len(y)): l = self.a * y[t] + (1 - self.a) * l
        self.level = l; return self
    def forecast(self, h):
        return np.full(h, self.level)

class Holt:
    """Double exponential smoothing (level + linear trend); alpha,beta tuned by SSE."""
    name = "Holt(trend)"
    def _run(self, y, a, b):
        l, tr, sse = y[0], y[1] - y[0], 0.0
        for t in range(1, len(y)):
            sse += (y[t] - (l + tr)) ** 2
            l_new = a * y[t] + (1 - a) * (l + tr)
            tr = b * (l_new - l) + (1 - b) * tr
            l = l_new
        return sse, l, tr
    def fit(self, y):
        best = (np.inf, 0.5, 0.1)
        for a in np.linspace(0.1, 0.9, 9):
            for b in np.linspace(0.05, 0.6, 6):
                sse = self._run(y, a, b)[0]
                if sse < best[0]: best = (sse, a, b)
        _, self.level, self.trend = self._run(y, best[1], best[2]); return self
    def forecast(self, h):
        return self.level + self.trend * np.arange(1, h + 1)

class HoltWinters:
    """Triple exponential smoothing (additive trend + season); a,b,g tuned by SSE."""
    def __init__(self, m): self.m = m; self.name = f"HoltWinters(m={m})"
    def _run(self, y, a, b, g):
        m = self.m
        l = y[:m].mean(); tr = (y[m:2*m].mean() - y[:m].mean()) / m
        s = [y[i] - l for i in range(m)]
        sse = 0.0
        for t in range(len(y)):
            si = t % m
            sse += (y[t] - (l + tr + s[si])) ** 2
            l_new = a * (y[t] - s[si]) + (1 - a) * (l + tr)
            tr = b * (l_new - l) + (1 - b) * tr
            s[si] = g * (y[t] - l_new) + (1 - g) * s[si]
            l = l_new
        return sse, l, tr, s
    def fit(self, y):
        best = (np.inf, 0.3, 0.1, 0.3)
        for a in (0.1, 0.3, 0.6):
            for b in (0.02, 0.1, 0.3):
                for g in (0.1, 0.3, 0.6):
                    sse = self._run(y, a, b, g)[0]
                    if sse < best[0]: best = (sse, a, b, g)
        _, self.level, self.trend, self.s = self._run(y, *best[1:]); self.n = len(y); return self
    def forecast(self, h):
        m = self.m
        return np.array([self.level + self.trend * k + self.s[(self.n + k - 1) % m]
                         for k in range(1, h + 1)])

class AR:
    """Autoregression of order p via least squares on lags; recursive multi-step."""
    def __init__(self, p=12): self.p = p; self.name = f"AR(p={p})"
    def fit(self, y):
        p = self.p
        X = np.array([y[t - p:t][::-1] for t in range(p, len(y))])
        X = np.hstack([np.ones((len(X), 1)), X])          # intercept column
        target = y[p:]
        self.w = np.linalg.lstsq(X, target, rcond=None)[0]
        self.hist = list(y[-p:]); return self
    def forecast(self, h):
        p, hist, out = self.p, list(self.hist), []
        for _ in range(h):
            lags = np.array(hist[-p:][::-1])
            pred = self.w[0] + self.w[1:] @ lags
            out.append(pred); hist.append(pred)
        return np.array(out)

class AutoForecaster:
    """Fit every candidate, score each on a held-out validation tail, keep the best."""
    def __init__(self, candidates, val_len=40):
        self.candidates, self.val_len = candidates, val_len
    def fit(self, y):
        cut = len(y) - self.val_len
        inner, val = y[:cut], y[cut:]
        self.scores = []
        for c in self.candidates:
            c.fit(inner)
            self.scores.append((rmse(c.forecast(self.val_len), val), c.name))
        self.scores.sort()
        self.best_name = self.scores[0][1]
        self.best = next(c for c in self.candidates if c.name == self.best_name)
        self.best.fit(y)                                   # refit winner on full history
        return self
    def forecast(self, h):
        return self.best.forecast(h)

def make_series(n=340, seed=0):
    # Trend + two seasonal cycles + noise -> recoverable structure for the auto-selector.
    np.random.seed(seed)
    t = np.arange(n)
    return 0.05 * t + 6.0 * np.sin(2*np.pi*t/25) + 2.0 * np.sin(2*np.pi*t/7) \
        + np.random.randn(n) * 0.4

if __name__ == "__main__":
    np.random.seed(0)
    series = make_series()
    test_len = 40
    train, test = series[:-test_len], series[-test_len:]   # forecast held-out tail

    m = detect_period(train)
    candidates = [NaiveLast(), MovingAverage(8), SES(), Holt(),
                  HoltWinters(m), AR(p=14)]
    auto = AutoForecaster(candidates, val_len=40).fit(train)

    pred = auto.forecast(test_len)
    naive = np.full(test_len, train[-1])                   # last-value baseline
    mean_b = np.full(test_len, train.mean())               # mean baseline

    print("Automated Time Series Forecasting System")
    print(f"  series={len(series)}  test_tail={test_len}  auto-detected period m={m}")
    print("  validation RMSE per candidate (lower = better):")
    for r, nm in auto.scores:
        star = "  <-- selected" if nm == auto.best_name else ""
        print(f"    {nm:>18s}: {r:7.3f}{star}")
    print(f"  SELECTED MODEL: {auto.best_name}")
    print(f"  naive (last-value) test RMSE: {rmse(naive, test):7.3f}   MAE: {np.mean(np.abs(naive-test)):.3f}")
    print(f"  mean             test RMSE: {rmse(mean_b, test):7.3f}   MAE: {np.mean(np.abs(mean_b-test)):.3f}")
    print(f"  AUTO forecast    test RMSE: {rmse(pred, test):7.3f}   MAE: {np.mean(np.abs(pred-test)):.3f}")
    print(f"  RMSE improvement over naive: {100*(1 - rmse(pred,test)/rmse(naive,test)):.1f}%")
    print(f"  result: {'AUTO BEATS naive + mean baselines' if rmse(pred,test) < min(rmse(naive,test), rmse(mean_b,test)) else 'baseline wins'}")
    print("  sample multi-step forecasts vs truth:")
    for j in range(0, test_len, max(1, test_len // 6)):
        print(f"    step {j+1:2d}  pred={pred[j]:7.3f}   true={test[j]:7.3f}   naive={naive[j]:7.3f}")
