import numpy as np

# Bitcoin Price Prediction Model (from scratch)
# ---------------------------------------------
# BTC price is nearly a random walk, so regressing on the raw price only relearns
# "tomorrow ~= today". Instead we predict the next-day LOG-RETURN from features that
# are actually knowable ahead of time: past returns (autoregression), a trailing
# EWMA-momentum, trailing volatility, AND calendar seasonality (crypto trades 7d/wk,
# so day-of-week carries a weekly cycle). We fit a ridge linear model via the manual
# normal equations, then reconstruct price as price_hat[t+1] = price[t]*exp(ret_hat).
# Beating a random-walk (last-value) baseline on price RMSE proves we recovered real
# structure in the returns. All math is by hand (only np.linalg.solve for the solve).


class BitcoinPricePredictor:
    """Ridge linear model on return + calendar features that forecasts next-day return."""

    def __init__(self, n_lags=6, ewma_alpha=0.3, vol_window=10, period=7, ridge=1e-3):
        self.n_lags = n_lags            # past log-returns used as AR features
        self.ewma_alpha = ewma_alpha    # smoothing for the momentum feature
        self.vol_window = vol_window    # trailing volatility (std of returns)
        self.period = period            # weekly seasonal cycle length
        self.ridge = ridge              # L2 penalty stabilizes the normal equations
        self.w = None
        self.mu = self.sd = None        # standardization stats for numeric features
        self._start = max(n_lags, vol_window)

    @staticmethod
    def _returns(prices):
        return np.diff(np.log(np.asarray(prices, float)))   # R[k]: price[k]->price[k+1]

    def _features(self, R, ks):
        # One row per target return R[k], using ONLY info available at time k.
        ks = np.asarray(ks)
        lags = np.stack([R[ks - j] for j in range(1, self.n_lags + 1)], axis=1)
        # EWMA momentum: exponentially weighted average of the last n_lags returns.
        wts = (1 - self.ewma_alpha) ** np.arange(self.n_lags)
        mom = (lags * wts).sum(1, keepdims=True) / wts.sum()
        win = np.stack([R[ks - i] for i in range(1, self.vol_window + 1)], axis=1)
        vol = np.std(win, axis=1, keepdims=True)            # trailing volatility
        num = np.hstack([lags, mom, vol])                   # numeric block
        # Day-of-week one-hot (drop last col to avoid collinearity with the bias).
        dow = ks % self.period
        onehot = (dow[:, None] == np.arange(self.period - 1)).astype(float)
        return num, onehot

    def _design(self, R, ks, fit=False):
        num, onehot = self._features(R, ks)
        if fit:
            self.mu, self.sd = num.mean(0), num.std(0) + 1e-12
        num_s = (num - self.mu) / self.sd
        return np.hstack([np.ones((len(num), 1)), num_s, onehot])   # bias + std num + dow

    def fit(self, prices):
        R = self._returns(prices)
        ks = np.arange(self._start, len(R))          # every target with full history
        X = self._design(R, ks, fit=True)
        y = R[ks]                                    # target = next-day log-return
        A = X.T @ X
        reg = self.ridge * np.eye(X.shape[1]); reg[0, 0] = 0.0        # do not penalize bias
        self.w = np.linalg.solve(A + reg, X.T @ y)
        return self

    def predict_return(self, prices, ks):
        R = self._returns(prices)
        return self._design(R, ks) @ self.w

    def predict_price(self, prices, ks):
        # One-step-ahead price: known price[k] scaled by predicted return -> price[k+1].
        prices = np.asarray(prices, float)
        return prices[np.asarray(ks)] * np.exp(self.predict_return(prices, ks))


def make_bitcoin_series(n=560, seed=0):
    # Geometric process whose LOG-RETURNS carry recoverable structure: an AR(2) piece
    # (momentum + short mean-reversion), a planted WEEKLY seasonal bump, small drift,
    # and noise. price = 20000 * exp(cumsum(returns)) -- BTC-like scale, not a pure
    # random walk, so a model that reads returns can genuinely beat last-value.
    np.random.seed(seed)
    drift = 0.0006
    seasonal = np.array([0.004, 0.001, -0.002, -0.003, 0.000, 0.002, -0.002])  # per weekday
    r = np.zeros(n)
    for i in range(2, n):
        r[i] = (drift + 0.45 * r[i - 1] - 0.25 * r[i - 2]
                + seasonal[i % 7] + 0.012 * np.random.randn())
    return 20000.0 * np.exp(np.cumsum(r))


if __name__ == "__main__":
    np.random.seed(0)
    prices = make_bitcoin_series(n=560)
    test_len = 100
    split = len(prices) - test_len                # forecast the held-out tail

    model = BitcoinPricePredictor(n_lags=6, ewma_alpha=0.3, vol_window=10, period=7)
    model.fit(prices[:split])

    # One-step-ahead over the held-out tail. Each prediction uses only true past prices
    # (index k) plus the known calendar day; it never peeks at the target price[k+1].
    R = BitcoinPricePredictor._returns(prices)
    ks = np.arange(split - 1, len(R))             # targets landing in the test tail
    pred_price = model.predict_price(prices, ks)
    true_price = prices[ks + 1]
    naive_price = prices[ks]                       # random-walk baseline: last value
    mean_price = np.full_like(true_price, prices[:split].mean())  # train-mean baseline

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, n_rmse, mean_rmse = rmse(pred_price, true_price), rmse(naive_price, true_price), rmse(mean_price, true_price)
    m_mae, n_mae = mae(pred_price, true_price), mae(naive_price, true_price)

    # Directional accuracy: did we call the up/down move right? Random baseline = 50%.
    pred_up = model.predict_return(prices, ks) > 0
    true_up = R[ks] > 0
    dir_acc = float(np.mean(pred_up == true_up))

    print("Bitcoin price prediction (ridge model on AR returns + momentum/vol + weekly seasonality)")
    print(f"  prices={len(prices)}  train={split}  test_tail={test_len}  features={len(model.w)}")
    print(f"  baseline train-mean   price RMSE: {mean_rmse:.2f}")
    print(f"  baseline random-walk  price RMSE: {n_rmse:.2f}   MAE: {n_mae:.2f}")
    print(f"  model  one-step       price RMSE: {m_rmse:.2f}   MAE: {m_mae:.2f}")
    print(f"  RMSE improvement over random-walk: {100 * (1 - m_rmse / n_rmse):.1f}%")
    print(f"  MAE  improvement over random-walk: {100 * (1 - m_mae / n_mae):.1f}%")
    print(f"  directional accuracy: {dir_acc:.3f}   (random baseline 0.500)")
    beat = m_rmse < n_rmse and m_rmse < mean_rmse and dir_acc > 0.5
    print(f"  result: {'model BEATS random-walk + mean baselines' if beat else 'baseline wins'}")
    print("  sample one-step forecasts vs truth (held-out tail):")
    for j in range(0, len(true_price), max(1, len(true_price) // 6)):
        print(f"    pred={pred_price[j]:10.2f}   true={true_price[j]:10.2f}   naive={naive_price[j]:10.2f}")
