import numpy as np

# Gold Price Prediction Model (from scratch)
# ------------------------------------------
# Gold (monthly) carries a slow upward TREND, an annual SEASONAL pattern, and
# mean-reverting deviations. We forecast by classic seasonal DECOMPOSITION,
# all math by hand:
#   1) linear TREND      price ~ a + b*t          (least squares on time)
#   2) additive SEASONAL s[t mod 12]              (avg detrended per month)
#   3) AR(p) on RESIDUAL resid[t] ~ sum phi*resid (normal equations, ridged)
# One-step forecast = trend(t) + seasonal(t) + AR(residual). A naive last-value
# baseline ignores trend+season and always lags, so beating it on the held-out
# tail proves the decomposition recovered the planted structure.


class GoldPricePredictor:
    """Trend + additive seasonality + AR(p) residual model, fit from scratch."""

    def __init__(self, period=12, ar_order=3, ridge=1e-6):
        self.period = period      # seasonal cycle length (12 = annual on monthly data)
        self.ar_order = ar_order  # AR lags used on the leftover residual
        self.ridge = ridge        # L2 stabilizer for the AR normal equations

    def fit(self, prices):
        y = np.asarray(prices, float)
        n = len(y)
        t = np.arange(n)

        # 1) linear trend via least squares: y ~ a + b*t
        A = np.vstack([np.ones(n), t]).T
        self.a, self.b = np.linalg.solve(A.T @ A, A.T @ y)
        detrended = y - (self.a + self.b * t)

        # 2) additive seasonal index per phase, centered to sum ~0
        self.season = np.array([detrended[t % self.period == k].mean()
                                for k in range(self.period)])
        self.season -= self.season.mean()
        resid = detrended - self.season[t % self.period]

        # 3) AR(p) on residual: resid[t] ~ sum_{j=1..p} phi_j * resid[t-j]
        p = self.ar_order
        rows = np.stack([resid[p - j - 1:n - j - 1] for j in range(p)], axis=1)
        target = resid[p:]
        G = rows.T @ rows + self.ridge * np.eye(p)
        self.phi = np.linalg.solve(G, rows.T @ target)
        return self

    def _residual(self, y, idx):
        # True residual at absolute indices idx (needs actual price[idx]).
        idx = np.asarray(idx)
        return y[idx] - (self.a + self.b * idx) - self.season[idx % self.period]

    def predict(self, prices, ks):
        # One-step-ahead price for targets ks, using ONLY info up to ks-1.
        y = np.asarray(prices, float)
        ks = np.asarray(ks)
        p = self.ar_order
        # deterministic trend + seasonal at the target time
        base = self.a + self.b * ks + self.season[ks % self.period]
        # AR term from the p true past residuals (no peeking at price[ks])
        ar = np.zeros(len(ks), float)
        for j in range(p):
            ar += self.phi[j] * self._residual(y, ks - j - 1)
        return base + ar


def make_gold_series(n=240, seed=0):
    # Monthly gold-like series: rising trend + annual seasonality + AR(1)
    # mean-reverting residual + small noise. Structure is recoverable (unlike a
    # pure random walk), yet a last-value baseline always lags the trend/season.
    np.random.seed(seed)
    t = np.arange(n)
    trend = 300.0 + 5.0 * t                                   # steady climb
    season = 40.0 * np.sin(2 * np.pi * t / 12.0) \
        + 18.0 * np.cos(2 * np.pi * t / 6.0)                  # annual + semiannual
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = 0.6 * resid[i - 1] + 12.0 * np.random.randn()  # AR(1) deviations
    return trend + season + resid


if __name__ == "__main__":
    np.random.seed(0)
    prices = make_gold_series(n=240)
    test_len = 48
    split = len(prices) - test_len               # forecast the held-out tail

    model = GoldPricePredictor(period=12, ar_order=3).fit(prices[:split])

    ks = np.arange(split, len(prices))           # one-step targets in the test tail
    pred = model.predict(prices, ks)
    true = prices[ks]
    naive = prices[ks - 1]                        # baseline: last observed value
    mean_base = np.full_like(true, prices[:split].mean())     # train-mean baseline

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, n_rmse, mean_rmse = rmse(pred, true), rmse(naive, true), rmse(mean_base, true)
    m_mae, n_mae = mae(pred, true), mae(naive, true)

    # Directional accuracy: did we call the up/down move right? Random = 50%.
    dir_acc = float(np.mean((pred > naive) == (true > naive)))

    print("Gold price prediction (trend + seasonality + AR residual decomposition)")
    print(f"  months={len(prices)}  train={split}  test_tail={test_len}  ar_order={model.ar_order}")
    print(f"  fitted trend: price ~ {model.a:.1f} + {model.b:.2f}*t")
    print(f"  baseline train-mean  RMSE: {mean_rmse:8.3f}")
    print(f"  baseline last-value  RMSE: {n_rmse:8.3f}   MAE: {n_mae:8.3f}")
    print(f"  model  one-step      RMSE: {m_rmse:8.3f}   MAE: {m_mae:8.3f}")
    print(f"  RMSE improvement over last-value: {100 * (1 - m_rmse / n_rmse):.1f}%")
    print(f"  MAE  improvement over last-value: {100 * (1 - m_mae / n_mae):.1f}%")
    print(f"  directional accuracy: {dir_acc:.3f}   (random baseline 0.500)")
    beat = m_rmse < n_rmse and m_rmse < mean_rmse and dir_acc > 0.5
    print(f"  result: {'model BEATS last-value + mean baselines' if beat else 'baseline wins'}")
    print("  sample one-step forecasts vs truth (held-out tail):")
    for j in range(0, len(true), max(1, len(true) // 6)):
        print(f"    pred={pred[j]:9.2f}   true={true[j]:9.2f}   naive={naive[j]:9.2f}")
