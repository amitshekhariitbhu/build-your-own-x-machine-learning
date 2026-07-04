import numpy as np

class ARIMA:
    """ARIMA(p,d,q) from scratch: difference d times, then fit AR + MA on the
    stationary series via the Hannan-Rissanen two-stage least-squares method."""
    def __init__(self, p=1, d=1, q=1):
        self.p, self.d, self.q = p, d, q

    def _difference(self, x):
        # Apply the (1-B)^d differencing operator to make the series stationary.
        for _ in range(self.d):
            x = np.diff(x)
        return x

    def _undiff(self, obs, w_next):
        # Invert d-fold differencing: rebuild a level value from the d-th diff.
        if self.d == 0:
            return w_next
        levels, cur = [obs], obs
        for _ in range(1, self.d):
            cur = np.diff(cur)
            levels.append(cur)
        val = w_next
        for k in range(self.d - 1, -1, -1):
            val = levels[k][-1] + val               # cumulate last value at each level
        return val

    def _residuals(self, w):
        # In-sample innovations e_t = w_t - model(w_{<t}, e_{<t}); MA makes this recursive.
        p, q = self.p, self.q
        e = np.zeros(len(w))
        for t in range(len(w)):
            pred = self.c
            for i in range(1, p + 1):
                if t - i >= 0:
                    pred += self.phi[i - 1] * w[t - i]
            for j in range(1, q + 1):
                if t - j >= 0:
                    pred += self.theta[j - 1] * e[t - j]
            e[t] = w[t] - pred
        return e

    def _ar_innovations(self, w, m):
        # Stage 1: a long AR(m) OLS fit gives proxy innovations to seed the MA part.
        n = len(w)
        X = np.ones((n - m, m + 1))
        for i in range(1, m + 1):
            X[:, i] = w[m - i:n - i]
        beta = np.linalg.lstsq(X, w[m:], rcond=None)[0]
        eps = np.zeros(n)
        eps[m:] = w[m:] - X @ beta
        return eps

    def fit(self, series):
        self.series = np.asarray(series, float)
        w = self._difference(self.series)
        p, q = self.p, self.q
        m = max(2 * (p + q), 10)                     # AR order for innovation estimate
        eps = self._ar_innovations(w, m)
        # Stage 2: regress w_t on its own p lags AND q lagged proxy-innovations.
        start = max(m + q, p)
        A, b = [], []
        for t in range(start, len(w)):
            row = [1.0]
            row += [w[t - i] for i in range(1, p + 1)]
            row += [eps[t - j] for j in range(1, q + 1)]
            A.append(row); b.append(w[t])
        beta = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
        self.c = beta[0]
        self.phi = beta[1:1 + p]                      # AR coefficients
        self.theta = beta[1 + p:1 + p + q]            # MA coefficients
        return self

    def _next_w(self, w, e):
        # One-step forecast of the differenced series from AR + MA terms.
        pred = self.c
        for i in range(1, self.p + 1):
            pred += self.phi[i - 1] * w[-i]
        for j in range(1, self.q + 1):
            pred += self.theta[j - 1] * e[-j]
        return pred

    def forecast_one(self, obs):
        # Predict the next raw value given observed history obs (params fixed).
        obs = np.asarray(obs, float)
        w = self._difference(obs)
        e = self._residuals(w)
        return self._undiff(obs, self._next_w(w, e))

    def forecast(self, steps):
        # Multi-step-ahead forecast from the end of training (future errors -> 0).
        obs = list(self.series)
        out = []
        for _ in range(steps):
            yhat = self.forecast_one(np.array(obs))
            out.append(yhat); obs.append(yhat)
        return np.array(out)

def make_series(n=360, seed=0):
    # Synthetic ARIMA(1,1,1)-style signal: integrate an ARMA(1,1) process and add drift.
    np.random.seed(seed)
    e = np.random.randn(n) * 0.5
    phi_t, theta_t = 0.6, 0.4
    w = np.zeros(n)                                   # stationary ARMA(1,1) increments
    for t in range(1, n):
        w[t] = phi_t * w[t - 1] + e[t] + theta_t * e[t - 1]
    return np.cumsum(w) + 0.05 * np.arange(n) + 10.0  # I(1) level with linear drift

if __name__ == "__main__":
    np.random.seed(0)
    series = make_series()
    test_len = 60
    split = len(series) - test_len                    # forecast the held-out tail

    model = ARIMA(p=1, d=1, q=1).fit(series[:split])

    # Rolling one-step-ahead forecasts: params fixed, state updated with observed truth.
    preds = np.array([model.forecast_one(series[:t]) for t in range(split, len(series))])
    true = series[split:]
    naive = series[split - 1:len(series) - 1]         # baseline: predict previous value

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    a_rmse, n_rmse = rmse(preds, true), rmse(naive, true)
    a_mae, n_mae = mae(preds, true), mae(naive, true)

    print("ARIMA(p=1,d=1,q=1) one-step forecasting (Hannan-Rissanen, from scratch)")
    print(f"  series={len(series)}  d=1 diff  test_tail={test_len}")
    print(f"  fitted: c={model.c:.3f}  phi={np.round(model.phi,3)}  theta={np.round(model.theta,3)}")
    print(f"  naive (last-value) RMSE: {n_rmse:.3f}   MAE: {n_mae:.3f}")
    print(f"  ARIMA forecast     RMSE: {a_rmse:.3f}   MAE: {a_mae:.3f}")
    print(f"  RMSE improvement over baseline: {100 * (1 - a_rmse / n_rmse):.1f}%")
    print(f"  MAE  improvement over baseline: {100 * (1 - a_mae / n_mae):.1f}%")
    print(f"  result: {'ARIMA BEATS naive baseline' if a_rmse < n_rmse else 'baseline wins'}")
    print("  sample forecasts vs truth (held-out tail):")
    for j in range(0, len(true), max(1, len(true) // 6)):
        print(f"    pred={preds[j]:8.3f}   true={true[j]:8.3f}   naive={naive[j]:8.3f}")
