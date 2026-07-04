import numpy as np

# Stock Price Prediction Model (from scratch)
# -------------------------------------------
# Stocks are close to a random walk, so we do NOT regress on raw price (that just
# relearns "tomorrow ~= today"). Instead we predict the next-day LOG-RETURN from
# engineered technical features (lagged returns, momentum, volatility) via ridge
# least squares (manual normal equations), then reconstruct the price as
#   price_hat[t+1] = price[t] * exp(return_hat).
# Beating a random-walk (last-value) baseline on price RMSE therefore means the
# model found real predictable structure in the returns. All math is by hand.


class StockPricePredictor:
    """Ridge linear model on technical features that predicts next-day log-return."""

    def __init__(self, n_lags=5, mom_windows=(3, 10), vol_window=10, ridge=1e-4):
        self.n_lags = n_lags            # past returns used as autoregressive features
        self.mom_windows = mom_windows  # momentum = summed returns over these windows
        self.vol_window = vol_window    # trailing volatility (std of returns)
        self.ridge = ridge              # L2 penalty stabilizes the normal equations
        self.w = None
        self.mu = self.sd = None        # feature standardization stats
        self._start = max(n_lags, max(mom_windows), vol_window)

    @staticmethod
    def _returns(prices):
        return np.diff(np.log(np.asarray(prices, float)))  # R[k]: price[k]->price[k+1]

    def _features(self, R, ks):
        # Row per target return R[k], using ONLY info available at time k (<= R[k-1]).
        ks = np.asarray(ks)
        lags = np.stack([R[ks - j] for j in range(1, self.n_lags + 1)], axis=1)
        mom = np.stack([np.sum(np.stack([R[ks - i] for i in range(1, w + 1)], 1), 1)
                        for w in self.mom_windows], axis=1)                 # momentum
        win = np.stack([R[ks - i] for i in range(1, self.vol_window + 1)], axis=1)
        vol = np.std(win, axis=1, keepdims=True)                            # volatility
        return np.hstack([lags, mom, vol])

    def fit(self, prices):
        R = self._returns(prices)
        ks = np.arange(self._start, len(R))          # every target with full history
        X = self._features(R, ks)
        y = R[ks]                                    # target = next-day log-return
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Xs = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])  # +bias col
        A = Xs.T @ Xs
        reg = self.ridge * np.eye(Xs.shape[1]); reg[0, 0] = 0.0           # free bias
        self.w = np.linalg.solve(A + reg, Xs.T @ y)
        return self

    def predict_return(self, prices, ks):
        R = self._returns(prices)
        X = self._features(R, ks)
        Xs = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        return Xs @ self.w

    def predict_price(self, prices, ks):
        # One-step-ahead price: known price[k] scaled by predicted return -> price[k+1].
        prices = np.asarray(prices, float)
        return prices[np.asarray(ks)] * np.exp(self.predict_return(prices, ks))


def make_stock_series(n=520, seed=0):
    # Geometric process whose LOG-RETURNS carry planted AR(2) structure (momentum +
    # short-term mean reversion) on top of a small drift: recoverable, unlike a pure
    # random walk. price = 100 * exp(cumsum(returns)).
    np.random.seed(seed)
    drift = 0.0004
    r = np.zeros(n)
    for i in range(2, n):
        r[i] = drift + 0.55 * r[i - 1] - 0.30 * r[i - 2] + 0.008 * np.random.randn()
    return 100.0 * np.exp(np.cumsum(r))


if __name__ == "__main__":
    np.random.seed(0)
    prices = make_stock_series(n=520)
    test_len = 100
    split = len(prices) - test_len               # forecast the held-out tail

    model = StockPricePredictor(n_lags=5, mom_windows=(3, 10), vol_window=10)
    model.fit(prices[:split])

    # One-step-ahead over the held-out tail. Each prediction uses only true past
    # prices (index k) and never peeks at the target price[k+1].
    R = StockPricePredictor._returns(prices)
    ks = np.arange(split - 1, len(R))            # targets landing in the test tail
    pred_price = model.predict_price(prices, ks)
    true_price = prices[ks + 1]
    naive_price = prices[ks]                      # random-walk baseline: last value
    mean_price = np.full_like(true_price, prices[:split].mean())  # train-mean baseline

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, n_rmse, mean_rmse = rmse(pred_price, true_price), rmse(naive_price, true_price), rmse(mean_price, true_price)
    m_mae, n_mae = mae(pred_price, true_price), mae(naive_price, true_price)

    # Directional accuracy: did we call the up/down move right? Baseline = 50%.
    pred_up = model.predict_return(prices, ks) > 0
    true_up = R[ks] > 0
    dir_acc = float(np.mean(pred_up == true_up))

    print("Stock price prediction (ridge model on returns/momentum/volatility features)")
    print(f"  prices={len(prices)}  train={split}  test_tail={test_len}  features={len(model.w)}")
    print(f"  baseline train-mean   price RMSE: {mean_rmse:.3f}")
    print(f"  baseline random-walk  price RMSE: {n_rmse:.3f}   MAE: {n_mae:.3f}")
    print(f"  model  one-step       price RMSE: {m_rmse:.3f}   MAE: {m_mae:.3f}")
    print(f"  RMSE improvement over random-walk: {100 * (1 - m_rmse / n_rmse):.1f}%")
    print(f"  MAE  improvement over random-walk: {100 * (1 - m_mae / n_mae):.1f}%")
    print(f"  directional accuracy: {dir_acc:.3f}   (random baseline 0.500)")
    beat = m_rmse < n_rmse and m_rmse < mean_rmse and dir_acc > 0.5
    print(f"  result: {'model BEATS random-walk + mean baselines' if beat else 'baseline wins'}")
    print("  sample one-step forecasts vs truth (held-out tail):")
    for j in range(0, len(true_price), max(1, len(true_price) // 6)):
        print(f"    pred={pred_price[j]:8.3f}   true={true_price[j]:8.3f}   naive={naive_price[j]:8.3f}")
