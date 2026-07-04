import numpy as np

# COVID-19 Case Prediction Model (from scratch)
# ---------------------------------------------
# Daily reported cases are an epidemic curve (several waves) riding on a strong WEEKLY
# reporting cycle: labs/health-depts report far fewer cases on weekends, so raw counts
# swing up/down 7-day-periodically. A last-value ("tomorrow == today") baseline gets
# punished at every Fri->Sat drop and Sun->Mon spike. We instead forecast the next day's
# LOG of new cases from features knowable in advance: past log-counts (autoregression),
# a trailing growth-momentum term (EWMA of daily log-growth = an effective-R proxy), and
# a day-of-week one-hot for the target day (its weekday is always known). We fit a ridge
# linear model via the manual normal equations, then map back with cases = exp(pred).
# Beating last-value AND train-mean RMSE on a held-out tail proves we recovered the wave
# dynamics + weekly seasonality. All math is by hand (np.linalg.solve only for the solve).


class CovidCasePredictor:
    """Ridge autoregression on log-cases + growth momentum + weekly seasonality."""

    def __init__(self, n_lags=7, ewma_alpha=0.35, period=7, ridge=1e-2):
        self.n_lags = n_lags          # past log-counts used as AR features
        self.ewma_alpha = ewma_alpha  # smoothing for the growth-momentum feature
        self.period = period          # weekly reporting cycle length
        self.ridge = ridge            # L2 penalty stabilizes the normal equations
        self.w = None
        self.mu = self.sd = None      # standardization stats for the numeric block

    def _features(self, logc, ts):
        # One row per target day t (predicting logc[t]); uses ONLY info from days < t.
        ts = np.asarray(ts)
        lags = np.stack([logc[ts - j] for j in range(1, self.n_lags + 1)], axis=1)
        # Growth momentum: EWMA of recent daily log-growth (log g[k] = logc[k]-logc[k-1]).
        growth = np.stack([logc[ts - j] - logc[ts - j - 1]
                           for j in range(1, self.n_lags + 1)], axis=1)
        wts = (1 - self.ewma_alpha) ** np.arange(self.n_lags)
        mom = (growth * wts).sum(1, keepdims=True) / wts.sum()
        num = np.hstack([lags, mom])                          # numeric block
        # Day-of-week one-hot for the TARGET day (drop last col to avoid bias collinearity).
        dow = ts % self.period
        onehot = (dow[:, None] == np.arange(self.period - 1)).astype(float)
        return num, onehot

    def _design(self, logc, ts, fit=False):
        num, onehot = self._features(logc, ts)
        if fit:
            self.mu, self.sd = num.mean(0), num.std(0) + 1e-12
        num_s = (num - self.mu) / self.sd
        return np.hstack([np.ones((len(num), 1)), num_s, onehot])   # bias + std num + dow

    def fit(self, cases):
        logc = np.log(np.asarray(cases, float))
        ts = np.arange(self.n_lags + 1, len(logc))     # every day with full history
        X = self._design(logc, ts, fit=True)
        y = logc[ts]                                   # target = next-day log-cases
        A = X.T @ X
        reg = self.ridge * np.eye(X.shape[1]); reg[0, 0] = 0.0   # do not penalize bias
        self.w = np.linalg.solve(A + reg, X.T @ y)
        return self

    def predict(self, cases, ts):
        # One-step-ahead cases for target days ts, reading only true past counts.
        logc = np.log(np.asarray(cases, float))
        return np.exp(self._design(logc, np.asarray(ts)) @ self.w)


def make_covid_series(n=430, seed=0):
    # Synthetic daily reported cases = (multi-wave epidemic curve) * (weekly reporting
    # cycle) * (lognormal noise). Waves are smooth Gaussian bumps -> strongly autocorrelated
    # so AR features work; the weekly factor plants a recoverable 7-day seasonality that a
    # last-value baseline cannot track. Counts stay strictly positive for the log model.
    np.random.seed(seed)
    t = np.arange(n)
    bump = lambda c, h, w: h * np.exp(-0.5 * ((t - c) / w) ** 2)
    waves = (300.0                                   # endemic baseline
             + bump(70, 4200, 22)                    # wave 1
             + bump(180, 9000, 30)                   # wave 2 (larger)
             + bump(300, 6500, 26)                   # wave 3
             + bump(390, 5000, 20))                  # wave 4 onset
    weekly = np.array([1.18, 1.12, 1.05, 1.00, 0.95, 0.66, 0.55])  # Mon..Sun reporting mult
    factor = weekly[t % 7]
    noise = np.exp(0.05 * np.random.randn(n))        # ~5% multiplicative reporting noise
    return np.maximum(np.round(waves * factor * noise), 1.0)


if __name__ == "__main__":
    np.random.seed(0)
    cases = make_covid_series(n=430)
    test_len = 90
    split = len(cases) - test_len                    # forecast the held-out tail

    model = CovidCasePredictor(n_lags=7, ewma_alpha=0.35, period=7)
    model.fit(cases[:split])

    # One-step-ahead over the held-out tail. Each prediction of cases[t] uses only true
    # counts from days < t (plus t's known weekday); it never peeks at the target.
    ts = np.arange(split, len(cases))
    pred = model.predict(cases, ts)
    true = cases[ts]
    naive = cases[ts - 1]                             # last-value baseline
    mean_base = np.full_like(true, cases[:split].mean(), dtype=float)  # train-mean baseline

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, n_rmse, mean_rmse = rmse(pred, true), rmse(naive, true), rmse(mean_base, true)
    m_mae, n_mae = mae(pred, true), mae(naive, true)

    # Directional accuracy: did we call the day-over-day up/down move right? Random = 50%.
    pred_up = pred > cases[ts - 1]
    true_up = true > cases[ts - 1]
    dir_acc = float(np.mean(pred_up == true_up))

    print("COVID-19 case prediction (ridge AR on log-cases + growth momentum + weekly seasonality)")
    print(f"  days={len(cases)}  train={split}  test_tail={test_len}  features={len(model.w)}")
    print(f"  baseline train-mean   cases RMSE: {mean_rmse:8.1f}")
    print(f"  baseline last-value   cases RMSE: {n_rmse:8.1f}   MAE: {n_mae:8.1f}")
    print(f"  model  one-step       cases RMSE: {m_rmse:8.1f}   MAE: {m_mae:8.1f}")
    print(f"  RMSE improvement over last-value: {100 * (1 - m_rmse / n_rmse):.1f}%")
    print(f"  MAE  improvement over last-value: {100 * (1 - m_mae / n_mae):.1f}%")
    print(f"  directional accuracy: {dir_acc:.3f}   (random baseline 0.500)")
    beat = m_rmse < n_rmse and m_rmse < mean_rmse and dir_acc > 0.5
    print(f"  result: {'model BEATS last-value + mean baselines' if beat else 'baseline wins'}")
    print("  sample one-step forecasts vs truth (held-out tail):")
    for j in range(0, len(true), max(1, len(true) // 6)):
        print(f"    pred={pred[j]:8.0f}   true={true[j]:8.0f}   naive={naive[j]:8.0f}")
