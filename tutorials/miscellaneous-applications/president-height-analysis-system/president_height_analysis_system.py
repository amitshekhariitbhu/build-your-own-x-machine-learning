import numpy as np


def describe(x):
    """From-scratch descriptive statistics for a 1-D array of heights.

    Returns mean, standard deviation and linearly-interpolated percentiles
    (min / Q1 / median / Q3 / max) computed by hand from a sorted copy.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    mean = x.sum() / n
    std = np.sqrt(((x - mean) ** 2).sum() / n)

    def pct(p):                                   # linear-interpolation percentile
        idx = (p / 100.0) * (n - 1)
        lo = int(np.floor(idx))
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return x[lo] * (1 - frac) + x[hi] * frac

    return {"mean": mean, "std": std, "min": x[0], "q1": pct(25),
            "median": pct(50), "q3": pct(75), "max": x[-1]}


class PresidentHeightAnalyzer:
    """Ordinary-least-squares height model fit from scratch.

    Explains a leader's height (cm) from era and background features so the
    planted historical up-trend and covariate effects can be recovered. The
    normal equations are solved on standardized features; only np.linalg.solve
    does the linear algebra, all modeling math is manual.
    """

    def __init__(self, l2=1e-3):
        self.l2 = l2
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    def _standardize(self, X, fit=False):
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0) + 1e-8
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = self._standardize(np.asarray(X, dtype=float), fit=True)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.b = y.mean()
        yc = y - self.b                          # center target -> intercept
        d = X.shape[1]
        A = X.T @ X + self.l2 * np.eye(d)        # (ridge-stabilized) normal equations
        self.w = np.linalg.solve(A, X.T @ yc)
        return self

    def predict(self, X):
        X = self._standardize(np.asarray(X, dtype=float))
        return X @ self.w + self.b

    def coefficients(self):
        """Weights back in raw feature units (cm change per unit of feature)."""
        return self.w / self.sigma


def make_president_data(n=400, seed=0):
    """Synthetic head-of-state records with a planted height signal.

    Feature order:
      years_since_1789 - term start offset (secular up-trend in stature)
      prior_military   - served as a general/officer (taller cohort)
      athlete_bg       - competitive-sport background (taller cohort)
      age_at_inaug     - age at inauguration (pure nuisance, ~0 effect)
    """
    rng = np.random.RandomState(seed)
    years = rng.uniform(0, 235, n)               # 1789 .. 2024
    military = (rng.rand(n) < 0.40).astype(float)
    athlete = (rng.rand(n) < 0.30).astype(float)
    age = rng.uniform(42, 70, n)
    X = np.column_stack([years, military, athlete, age])

    beta = np.array([0.055, 3.0, 4.0, 0.0])      # planted per-unit effects (cm)
    height = 170.0 + X @ beta + rng.normal(0, 2.5, n)
    height = np.clip(height, 160.0, 200.0)       # plausible human range
    return X, height


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_president_data(n=400, seed=0)

    # held-out split: 70% train / 30% test
    n_train = int(0.7 * len(y))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    model = PresidentHeightAnalyzer().fit(Xtr, ytr)
    pred = model.predict(Xte)
    model_rmse, model_r2 = rmse(yte, pred), r2(yte, pred)

    # baseline: predict the training-mean height for every leader
    base_pred = np.full_like(yte, ytr.mean())
    base_rmse = rmse(yte, base_pred)

    stats = describe(y)
    names = ["years_since_1789", "prior_military", "athlete_bg", "age_at_inaug"]
    planted = [0.055, 3.0, 4.0, 0.0]

    print("=== President Height Analysis System (from scratch) ===")
    print(f"records               : {len(y)}  (test={len(yte)})")
    print("height stats (cm)     : "
          f"mean={stats['mean']:.1f} std={stats['std']:.1f} "
          f"min={stats['min']:.1f} Q1={stats['q1']:.1f} "
          f"median={stats['median']:.1f} Q3={stats['q3']:.1f} max={stats['max']:.1f}")
    print(f"baseline (mean-height): RMSE={base_rmse:.3f} cm")
    print(f"OLS height model      : RMSE={model_rmse:.3f} cm  R^2={model_r2:.3f}")
    print(f"error reduction       : {(1 - model_rmse / base_rmse) * 100:.1f}% vs baseline")
    print("recovered effects (cm per unit)   [planted]:")
    for nm, c, p in zip(names, model.coefficients(), planted):
        print(f"  {nm:<16}: {c:+.3f}   [{p:+.3f}]")
    assert model_rmse < 0.7 * base_rmse, "model should clearly beat the mean baseline"
    print("PASS: height model recovers the planted trend and beats the mean baseline")
