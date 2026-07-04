import numpy as np

# Fuel Efficiency Prediction Model via seasonal decomposition + residual autoregression, from scratch.
# A delivery van's monthly fuel economy (km per litre) carries three planted signals: a slow
# downward TREND (engine/tyre wear steadily eats efficiency), a 12-month SEASONAL cycle (cold
# winters hurt economy, mild summers help), and an AR(1) RESIDUAL giving month-to-month drag.
# Stage 1 fits the deterministic part -- a linear time trend plus month-of-year dummies -- by
# least squares. Stage 2 fits an AR(p) model on the leftover residuals. The held-out final year
# is forecast by extrapolating the trend+season and recursively rolling the AR residual forward,
# then scored against naive last-value, seasonal-naive (t-12), and mean baselines.


def make_fuel_efficiency(T=96, period=12, seed=0):
    # Synthetic monthly km/L with recoverable trend + seasonal + AR(1) structure.
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    month = t % period
    base, degrade = 13.5, -0.07                           # new-van economy + monthly wear
    # Jan..Dec seasonal offset (km/L): winter penalty, summer bonus.
    shape = np.array([-0.9, -0.7, -0.3, 0.2, 0.6, 0.9,
                      1.0, 0.8, 0.4, -0.1, -0.5, -0.8])
    season = shape[month]
    e = np.zeros(T)                                       # AR(1) residual momentum
    for i in range(1, T):
        e[i] = 0.6 * e[i - 1] + rng.normal(0, 0.10)
    return base + degrade * t + season + e


class SeasonalTrendAR:
    """Deterministic trend + monthly-dummy regression, with AR(p) on the residuals."""

    def __init__(self, period=12, ar_order=2):
        self.period, self.ar_order = period, ar_order

    def _design(self, t):
        # Deterministic feature row for absolute time index t: intercept, trend, month dummies.
        row = [1.0, t / 100.0]                            # intercept + scaled linear trend
        month = t % self.period
        onehot = [0.0] * (self.period - 1)               # month-of-year one-hot (drop last)
        if month < self.period - 1:
            onehot[month] = 1.0
        return row + onehot

    def fit(self, y):
        y = np.asarray(y, float)
        n = len(y)
        # Stage 1: least-squares fit of trend + seasonal dummies.
        X = np.array([self._design(t) for t in range(n)])
        self.det_coef_, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ self.det_coef_
        # Stage 2: AR(p) on the (near zero-mean) residuals via least squares.
        p = self.ar_order
        Xr = np.array([resid[t - p:t][::-1] for t in range(p, n)])  # cols = lag1..lagp
        self.ar_coef_, *_ = np.linalg.lstsq(Xr, resid[p:], rcond=None)
        self._resid, self._n = resid, n
        return self

    def forecast(self, steps):
        p = self.ar_order
        # Deterministic component extrapolated into the future.
        det = np.array([np.dot(self._design(self._n + i), self.det_coef_)
                        for i in range(steps)])
        # Recursive AR forecast of the residual, feeding predictions back as lags.
        hist = list(self._resid[-p:])
        r_preds = []
        for _ in range(steps):
            lags = np.array(hist[-p:][::-1])
            r = float(np.dot(self.ar_coef_, lags))
            r_preds.append(r)
            hist.append(r)
        return det + np.array(r_preds)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


if __name__ == "__main__":
    np.random.seed(0)
    period, horizon = 12, 12
    y = make_fuel_efficiency(T=96, period=period)
    train, test = y[:-horizon], y[-horizon:]             # hold out the final 12 months

    model = SeasonalTrendAR(period=period, ar_order=2).fit(train)
    pred = model.forecast(horizon)

    # Baselines: repeat last month; repeat same month a year ago; repeat the mean.
    naive = np.full(horizon, train[-1])
    seasonal_naive = np.array([train[-period + (i % period)] for i in range(horizon)])
    mean_base = np.full(horizon, train.mean())

    print("Fuel Efficiency Prediction Model -- seasonal decomposition + residual AR")
    print(f"train months={len(train)}  test months={horizon}  period={period}  ar_order=2")
    print(f"{'model':<26}{'RMSE':>9}{'MAE':>9}")
    print(f"{'trend+season+AR':<26}{rmse(pred, test):>9.3f}{mae(pred, test):>9.3f}")
    print(f"{'naive (last value)':<26}{rmse(naive, test):>9.3f}{mae(naive, test):>9.3f}")
    print(f"{'seasonal-naive (t-12)':<26}{rmse(seasonal_naive, test):>9.3f}"
          f"{mae(seasonal_naive, test):>9.3f}")
    print(f"{'mean':<26}{rmse(mean_base, test):>9.3f}{mae(mean_base, test):>9.3f}")

    m_r = rmse(pred, test)
    best_base = min(rmse(naive, test), rmse(seasonal_naive, test), rmse(mean_base, test))
    lift = 100.0 * (best_base - m_r) / best_base
    print(f"\nRMSE improvement over best baseline: {lift:.1f}%")
    print("RESULT:", "PASS -- beats every baseline" if m_r < best_base else "FAIL")
