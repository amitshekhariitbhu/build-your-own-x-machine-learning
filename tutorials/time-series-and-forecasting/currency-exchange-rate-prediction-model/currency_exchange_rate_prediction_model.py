import numpy as np

# Currency Exchange Rate Prediction Model (from scratch)
# ------------------------------------------------------
# FX rates behave like a random walk, so regressing on the raw rate just relearns
# "tomorrow ~= today". Instead we predict the next-day LOG-RETURN from features
# that capture the two forces real currencies show:
#   - MEAN REVERSION: rates drift back toward a slow-moving "fair value" (a proxy
#     for purchasing-power / interest-rate parity), so the deviation of the rate
#     from its trailing average predicts the coming move.
#   - MOMENTUM: short-term autocorrelation in returns (lagged returns).
# A ridge linear model (manual normal equations) maps these to the next return,
# then we rebuild the rate as  rate_hat[t+1] = rate[t] * exp(return_hat).
# Beating a random-walk (last-value) RMSE proves we recovered real structure.


class CurrencyRatePredictor:
    """Ridge linear model on mean-reversion + momentum features -> next log-return."""

    def __init__(self, n_lags=4, rev_window=20, mom_windows=(3, 10), vol_window=10, ridge=1e-3):
        self.n_lags = n_lags            # past returns used as autoregressive features
        self.rev_window = rev_window    # trailing window whose mean proxies fair value
        self.mom_windows = mom_windows  # momentum = summed returns over these windows
        self.vol_window = vol_window    # trailing volatility (std of returns)
        self.ridge = ridge              # L2 penalty stabilizes the normal equations
        self.w = None
        self.mu = self.sd = None        # feature standardization stats
        self._start = max(n_lags, rev_window, max(mom_windows), vol_window)

    def _features(self, logp, R, ks):
        # Row per target return R[k], using ONLY info available at time k (<= R[k-1]
        # and log-rates up to logp[k]); never peeks at the future.
        ks = np.asarray(ks)
        lags = np.stack([R[ks - j] for j in range(1, self.n_lags + 1)], axis=1)
        mom = np.stack([np.sum(np.stack([R[ks - i] for i in range(1, w + 1)], 1), 1)
                        for w in self.mom_windows], axis=1)                    # momentum
        win = np.stack([R[ks - i] for i in range(1, self.vol_window + 1)], axis=1)
        vol = np.std(win, axis=1, keepdims=True)                              # volatility
        ma = np.mean(np.stack([logp[ks - i] for i in range(self.rev_window)], 1), 1)
        rev = (logp[ks] - ma).reshape(-1, 1)          # deviation from trailing fair value
        return np.hstack([lags, mom, vol, rev])

    def fit(self, rates):
        logp = np.log(np.asarray(rates, float))
        R = np.diff(logp)                            # R[k]: log-rate[k] -> log-rate[k+1]
        ks = np.arange(self._start, len(R))          # every target with full history
        X = self._features(logp, R, ks)
        y = R[ks]                                    # target = next-day log-return
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Xs = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])   # +bias col
        A = Xs.T @ Xs
        reg = self.ridge * np.eye(Xs.shape[1]); reg[0, 0] = 0.0            # free bias
        self.w = np.linalg.solve(A + reg, Xs.T @ y)
        return self

    def predict_return(self, rates, ks):
        logp = np.log(np.asarray(rates, float))
        R = np.diff(logp)
        X = self._features(logp, R, ks)
        Xs = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        return Xs @ self.w

    def predict_rate(self, rates, ks):
        # One-step-ahead rate: known rate[k] scaled by predicted return -> rate[k+1].
        rates = np.asarray(rates, float)
        return rates[np.asarray(ks)] * np.exp(self.predict_return(rates, ks))


def make_fx_series(n=700, seed=0):
    # Exchange rate with PLANTED structure: an equilibrium "fair value" that drifts
    # on a slow cycle, plus returns driven by mean reversion toward it and AR(2)
    # momentum. rate = exp(log_rate); recoverable, unlike a pure random walk.
    np.random.seed(seed)
    t = np.arange(n)
    log_eq = np.log(1.20) + 0.08 * np.sin(2 * np.pi * t / 260.0)   # slow fair-value cycle
    logp = np.empty(n)
    logp[0] = log_eq[0]
    r = np.zeros(n)
    for i in range(1, n):
        rev = 0.05 * (log_eq[i - 1] - logp[i - 1])     # pull back toward fair value
        ar = 0.25 * r[i - 1] - 0.12 * (r[i - 2] if i >= 2 else 0.0)   # AR(2) momentum
        r[i] = rev + ar + 0.004 * np.random.randn()
        logp[i] = logp[i - 1] + r[i]
    return np.exp(logp)


if __name__ == "__main__":
    np.random.seed(0)
    rates = make_fx_series(n=700)
    test_len = 150
    split = len(rates) - test_len                 # forecast the held-out tail

    model = CurrencyRatePredictor(n_lags=4, rev_window=20, mom_windows=(3, 10)).fit(rates[:split])

    # One-step-ahead over the held-out tail. Each prediction uses only true past
    # rates (index k) and never peeks at the target rate[k+1].
    R = np.diff(np.log(rates))
    ks = np.arange(split - 1, len(R))             # targets landing in the test tail
    pred_rate = model.predict_rate(rates, ks)
    true_rate = rates[ks + 1]
    naive_rate = rates[ks]                         # random-walk baseline: last value
    mean_rate = np.full_like(true_rate, rates[:split].mean())      # train-mean baseline

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, n_rmse, mean_rmse = rmse(pred_rate, true_rate), rmse(naive_rate, true_rate), rmse(mean_rate, true_rate)
    m_mae, n_mae = mae(pred_rate, true_rate), mae(naive_rate, true_rate)

    # Directional accuracy: did we call the up/down move right? Baseline = 50%.
    pred_up = model.predict_return(rates, ks) > 0
    true_up = R[ks] > 0
    dir_acc = float(np.mean(pred_up == true_up))

    print("Currency exchange rate prediction (ridge model on mean-reversion + momentum)")
    print(f"  rates={len(rates)}  train={split}  test_tail={test_len}  features={len(model.w)}")
    print(f"  baseline train-mean   rate RMSE: {mean_rmse:.5f}")
    print(f"  baseline random-walk  rate RMSE: {n_rmse:.5f}   MAE: {n_mae:.5f}")
    print(f"  model  one-step       rate RMSE: {m_rmse:.5f}   MAE: {m_mae:.5f}")
    print(f"  RMSE improvement over random-walk: {100 * (1 - m_rmse / n_rmse):.1f}%")
    print(f"  MAE  improvement over random-walk: {100 * (1 - m_mae / n_mae):.1f}%")
    print(f"  directional accuracy: {dir_acc:.3f}   (random baseline 0.500)")
    beat = m_rmse < n_rmse and m_rmse < mean_rmse and dir_acc > 0.5
    print(f"  result: {'model BEATS random-walk + mean baselines' if beat else 'baseline wins'}")
    print("  sample one-step forecasts vs truth (held-out tail):")
    for j in range(0, len(true_rate), max(1, len(true_rate) // 6)):
        print(f"    pred={pred_rate[j]:8.5f}   true={true_rate[j]:8.5f}   naive={naive_rate[j]:8.5f}")
