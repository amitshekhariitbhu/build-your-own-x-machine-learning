import numpy as np

class MigrationForecaster:
    """Net-migration forecaster built from scratch with ridge regression (normal
    equations) on autoregressive lags plus KNOWN drivers a demographer can supply
    ahead of time:
      - recent net-migration lags (short memory + same-month-last-year),
      - Fourier harmonics for the yearly (12-month) seasonal moving cycle,
      - an economic 'pull' index = destination-minus-origin opportunity gap
        (jobs/wages); people move toward opportunity, so this drives inflow,
      - a policy/border-openness index that scales how freely people can move.
    The exogenous drivers are policy/economic scenarios known at forecast time, so
    a planner can ask 'what if the labor market tightens or the border opens?'."""
    def __init__(self, lags=(1, 2, 12, 13), period=12, n_harmonics=2, ridge=2.0):
        self.lags = tuple(lags)       # which past net-migration offsets feed each row
        self.period = period          # months per seasonal cycle
        self.n_harmonics = n_harmonics
        self.ridge = ridge            # L2 penalty stabilizes the least-squares solve
        self.w = None

    def _exog(self, t, pull, policy):
        # Known drivers at absolute month index t (all knowable ahead of forecast).
        t = np.atleast_1d(t).astype(float)
        pull = np.atleast_1d(pull).astype(float); policy = np.atleast_1d(policy).astype(float)
        feats = [np.ones_like(t), pull, policy, pull * policy]  # openness gates the pull
        for k in range(1, self.n_harmonics + 1):
            feats.append(np.sin(2 * np.pi * k * t / self.period))
            feats.append(np.cos(2 * np.pi * k * t / self.period))
        return np.stack(feats, axis=1)

    def _design(self, series, pull, policy, times):
        # One row per target month: [migration lags..., known exogenous drivers at t].
        L = np.stack([series[times - k] for k in self.lags], axis=1)
        return np.hstack([L, self._exog(times, pull[times], policy[times])])

    def fit(self, series, pull, policy):
        series = np.asarray(series, float)
        times = np.arange(max(self.lags), len(series))
        X = self._design(series, pull, policy, times); y = series[times]
        # Ridge normal equations: w = (X'X + lambda I)^-1 X'y (leave bias col unpenalized).
        A = X.T @ X
        reg = self.ridge * np.eye(X.shape[1]); reg[len(self.lags), len(self.lags)] = 0.0
        self.w = np.linalg.solve(A + reg, X.T @ y)
        return self

    def predict_onestep(self, series, pull, policy, times):
        # One-step-ahead net migration using TRUE recent history + known drivers at t.
        X = self._design(np.asarray(series, float), pull, policy, np.asarray(times))
        return X @ self.w

    def forecast(self, history, pull, policy, times):
        # Recursive multi-step: feed each prediction back in as a future lag.
        buf = list(np.asarray(history, float))
        out = []
        for t in np.asarray(times):
            lags = np.array([buf[t - k] for k in self.lags])
            row = np.hstack([lags, self._exog(t, pull[t], policy[t])[0]])
            yhat = float(row @ self.w)
            out.append(yhat)
            buf.append(yhat) if t == len(buf) else buf.__setitem__(t, yhat)
        return np.array(out)

def make_migration(n=180, period=12, seed=0):
    # Synthetic monthly net migration (thousands) with recoverable latent structure:
    #   base + slow growth + summer moving season + economic pull*openness + AR(1) noise.
    np.random.seed(seed)
    t = np.arange(n); month = t % period
    # Economic pull index: destination-minus-origin opportunity gap, mean-reverting walk.
    pull = np.zeros(n)
    for i in range(1, n):
        pull[i] = 0.9 * pull[i - 1] + np.random.randn() * 0.5
    pull = pull + 1.5 * np.sin(2 * np.pi * t / 60)          # slow economic super-cycle
    # Border/policy openness in [0.3, 1.0]: occasional tightening episodes.
    policy = 0.8 + 0.15 * np.sin(2 * np.pi * t / 48)
    policy[(t > 70) & (t < 95)] -= 0.35                     # a tightening episode
    policy = np.clip(policy, 0.3, 1.0)
    seasonal = 6.0 * np.sin(2 * np.pi * (month - 3) / period) \
        + 2.0 * np.sin(4 * np.pi * month / period)         # summer peak + secondary bump
    trend = 20.0 + 0.05 * t                                 # base inflow + slow growth
    noise = np.zeros(n)
    for i in range(1, n):                                   # AR(1) -> lags carry real signal
        noise[i] = 0.6 * noise[i - 1] + np.random.randn() * 1.2
    migration = trend + seasonal + 4.0 * pull * policy + noise
    return migration, pull, policy

if __name__ == "__main__":
    np.random.seed(0)
    period, test_len = 12, 36                               # forecast the held-out final 3 years
    mig, pull, policy = make_migration(n=180, period=period)
    split = len(mig) - test_len

    model = MigrationForecaster(lags=(1, 2, 12, 13), period=period, n_harmonics=2, ridge=2.0)
    model.fit(mig[:split], pull[:split], policy[:split])

    idx = np.arange(split, len(mig))                        # held-out month indices
    pred = model.predict_onestep(mig, pull, policy, idx)    # one-step-ahead (true past)
    multi = model.forecast(mig[:split], pull, policy, idx)  # pure recursive multi-step
    true = mig[idx]
    naive = mig[idx - 1]                                    # baseline: previous month
    season_naive = mig[idx - period]                        # baseline: same month last year
    mean_base = np.full_like(true, mig[:split].mean())      # baseline: train mean

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    m_rmse, m_mae = rmse(pred, true), mae(pred, true)
    mult_rmse = rmse(multi, true)
    bases = {"train-mean": mean_base, "last-month": naive, "same-month-last-year": season_naive}
    base_rmse = {k: rmse(v, true) for k, v in bases.items()}
    best_name = min(base_rmse, key=base_rmse.get); best = base_rmse[best_name]

    print("Migration prediction (ridge AR + yearly Fourier + economic pull * border openness)")
    print(f"  months={len(mig)}  lags={model.lags}  period={period}  test_tail={test_len}")
    for k, v in base_rmse.items():
        print(f"  baseline {k:<22s} RMSE: {v:7.3f}")
    print(f"  migration model one-step   RMSE: {m_rmse:7.3f}   MAE: {m_mae:7.3f}")
    print(f"  migration model multi-step RMSE: {mult_rmse:7.3f}   (recursive, no peeking)")
    print(f"  RMSE improvement over best baseline ({best_name}): {100*(1-m_rmse/best):.1f}%")
    print(f"  result: {'migration model BEATS all baselines' if m_rmse < best else 'baseline wins'}")
    print("  sample one-step forecasts vs truth (held-out months):")
    for j in range(0, len(true), max(1, len(true) // 6)):
        print(f"    month={idx[j]:3d}  pred={pred[j]:6.2f}  true={true[j]:6.2f}  "
              f"pull={pull[idx][j]:5.2f}  policy={policy[idx][j]:4.2f}")
