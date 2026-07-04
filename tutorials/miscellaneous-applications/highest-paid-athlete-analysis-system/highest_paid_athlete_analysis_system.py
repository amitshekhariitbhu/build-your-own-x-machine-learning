import numpy as np


class AthleteEarningsModel:
    """Ridge regression (closed form) that models athlete annual earnings.

    Earnings are heavily right-skewed, so the target is log10(earnings in $M).
    Features are standardized, then weights are solved with the ridge normal
    equation  w = (X^T X + lambda I)^-1 X^T y  (np.linalg allowed). predict()
    maps back to dollars; the same scores rank athletes to surface the
    highest-paid stars.
    """

    def __init__(self, l2=1.0):
        self.l2 = l2          # ridge penalty (stabilizes the solve)
        self.w = None
        self.mu = None
        self.sigma = None

    def _standardize(self, X, fit=False):
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0) + 1e-8
        return (X - self.mu) / self.sigma

    def fit(self, X, y_log):
        X = self._standardize(np.asarray(X, dtype=float), fit=True)
        y = np.asarray(y_log, dtype=float).reshape(-1)
        n, d = X.shape
        Xb = np.column_stack([np.ones(n), X])          # bias column
        A = Xb.T @ Xb + self.l2 * np.eye(d + 1)
        A[0, 0] -= self.l2                             # don't penalize bias
        self.w = np.linalg.solve(A, Xb.T @ y)
        return self

    def predict_log(self, X):
        X = self._standardize(np.asarray(X, dtype=float))
        Xb = np.column_stack([np.ones(X.shape[0]), X])
        return Xb @ self.w

    def predict(self, X):                              # earnings in $M
        return 10.0 ** self.predict_log(X)


def make_athlete_data(n=600, seed=0):
    """Synthetic athletes with a planted, recoverable earnings signal.

    Feature order:
      sport_market   - marketability of the athlete's sport (league TV money)
      social_millions- social-media following in millions (endorsement pull)
      performance    - normalized on-field performance / win share
      championships   - titles won (fame multiplier)
      endorsements   - number of active brand deals
      age            - career-arc; earnings peak in the late 20s
    log10(earnings $M) is a linear function of these (plus an age hump) with
    noise, so a big salary signal exists but individual deals are noisy.
    """
    rng = np.random.RandomState(seed)
    sport_market = rng.uniform(0, 1, n)
    social_millions = rng.gamma(2.0, 8.0, n)
    performance = rng.normal(0.0, 1.0, n)
    championships = rng.poisson(1.2, n)
    endorsements = rng.poisson(3.0, n)
    age = rng.uniform(19, 38, n)
    X = np.column_stack([sport_market, social_millions, performance,
                         championships, endorsements, age])

    # Generative log10-earnings (~$1M..$120M): endorsement reach dominates.
    log_earn = (0.9
                + 0.70 * sport_market
                + 0.020 * social_millions
                + 0.18 * performance
                + 0.12 * championships
                + 0.06 * endorsements
                - 0.006 * (age - 28.0) ** 2)          # peak near 28
    log_earn += rng.normal(0.0, 0.12, n)              # contract-to-contract noise
    return X, log_earn


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


if __name__ == "__main__":
    np.random.seed(0)

    X, y_log = make_athlete_data(n=600, seed=0)
    earn = 10.0 ** y_log                               # true earnings ($M)

    # Feature engineering: add age^2 so the linear model can fit the
    # career-arc hump (earnings rise, peak, then fade with age).
    X = np.column_stack([X, X[:, 5] ** 2])

    n_train = int(0.7 * len(y_log))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr_log, yte_log = y_log[:n_train], y_log[n_train:]
    earn_te = earn[n_train:]

    model = AthleteEarningsModel(l2=1.0).fit(Xtr, ytr_log)
    pred_log = model.predict_log(Xte)                  # predicted log10 earnings
    pred_te = model.predict(Xte)                       # predicted earnings ($M)

    # Baseline: predict every athlete earns the training-set mean.
    base_log = np.full_like(yte_log, ytr_log.mean())
    base_pred = np.full_like(earn_te, np.mean(10.0 ** ytr_log))

    # Primary metric = log10-earnings RMSE (what the model actually fits).
    model_log_rmse = rmse(pred_log, yte_log)
    base_log_rmse = rmse(base_log, yte_log)
    r2 = 1.0 - np.sum((yte_log - pred_log) ** 2) / np.sum((yte_log - yte_log.mean()) ** 2)
    # Secondary flavor metric = dollar RMSE (skewed by a few mega-earners).
    model_rmse = rmse(pred_te, earn_te)
    base_rmse = rmse(base_pred, earn_te)

    # Ranking task: recover the actual top-10 highest-paid from predictions.
    k = 10
    true_top = set(np.argsort(-earn_te)[:k])
    pred_top = set(np.argsort(-pred_te)[:k])
    hits = len(true_top & pred_top)
    rand_hits = k * k / len(earn_te)                   # expected random overlap

    print("=== Highest-Paid Athlete Analysis System (from scratch) ===")
    print(f"athletes: {len(y_log)}  (train {n_train} / test {len(yte_log)})")
    print(f"baseline log10-earn RMSE : {base_log_rmse:.3f}")
    print(f"ridge  log10-earn RMSE   : {model_log_rmse:.3f}")
    print(f"improvement over baseline: {(1 - model_log_rmse / base_log_rmse) * 100:4.1f}% lower RMSE")
    print(f"model R^2 (log earnings) : {r2:.3f}")
    print(f"dollar RMSE   base ${base_rmse:6.1f}M  ->  model ${model_rmse:6.1f}M")
    print(f"top-{k} highest-paid recovered : {hits}/{k}  (random ~{rand_hits:.1f})")
    print("richest predicted athlete    : "
          f"${pred_te.max():.1f}M   (actual ${earn_te.max():.1f}M)")

    assert model_log_rmse < 0.5 * base_log_rmse, "model should beat mean baseline"
    assert r2 > 0.8, "model should explain most log-earnings variance"
    assert hits >= 5, "model should recover most of the true top-10 earners"
    print("PASS: earnings model beats mean baseline and ranks top earners")
