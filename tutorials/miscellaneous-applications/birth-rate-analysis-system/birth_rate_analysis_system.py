import numpy as np


class BirthRateAnalyzer:
    """Ridge linear-regression birth-rate model from scratch.

    Predicts a region's crude birth rate (live births per 1000 people) from
    socioeconomic indicators. Fit in closed form via the normal equations with
    an L2 (ridge) penalty on standardized features; np.linalg.solve does the
    linear solve, all of the modeling math is manual.
    """

    def __init__(self, l2=1.0):
        self.l2 = l2      # ridge strength (0 = ordinary least squares)
        self.w = None     # standardized-space weights
        self.b = 0.0      # intercept (raw target mean)
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
        yc = y - self.b                       # center target -> intercept
        d = X.shape[1]
        A = X.T @ X + self.l2 * np.eye(d)     # ridge normal-equations matrix
        self.w = np.linalg.solve(A, X.T @ yc)
        return self

    def predict(self, X):
        X = self._standardize(np.asarray(X, dtype=float))
        return X @ self.w + self.b

    def coefficients(self):
        """Weights back in raw feature units (per-unit effect on birth rate)."""
        return self.w / self.sigma


def make_birth_data(n=600, seed=0):
    """Synthetic region-year records with a planted birth-rate signal.

    Feature order:
      female_edu_yrs   - mean years of schooling for women (up -> fewer births)
      gdp_k            - GDP per capita in $1000s          (up -> fewer births)
      urban_pct        - share of population in cities      (up -> fewer births)
      contracep_pct    - contraceptive prevalence           (up -> fewer births)
      infant_mort      - infant deaths per 1000 births      (up -> more  births)
      median_age       - population median age in years      (up -> fewer births)
    """
    rng = np.random.RandomState(seed)
    female_edu = rng.uniform(2, 14, n)
    gdp_k = rng.uniform(1, 70, n)
    urban = rng.uniform(15, 90, n)
    contracep = rng.uniform(5, 80, n)
    infant = rng.uniform(3, 90, n)
    median_age = rng.uniform(16, 45, n)
    X = np.column_stack([female_edu, gdp_k, urban, contracep, infant, median_age])

    # planted linear generating process (crude birth rate per 1000)
    beta = np.array([-1.10, -0.12, -0.06, -0.09, 0.16, -0.28])
    birth = 44.0 + X @ beta + rng.normal(0, 1.6, n)   # modest observation noise
    birth = np.clip(birth, 6.0, 50.0)                 # plausible real-world range
    return X, birth


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_birth_data(n=600, seed=0)

    # held-out split: 70% train / 30% test
    n_train = int(0.7 * len(y))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    model = BirthRateAnalyzer(l2=1.0).fit(Xtr, ytr)
    pred = model.predict(Xte)
    model_rmse, model_r2 = rmse(yte, pred), r2(yte, pred)

    # baseline: predict the training-mean birth rate for every region
    base_pred = np.full_like(yte, ytr.mean())
    base_rmse = rmse(yte, base_pred)

    names = ["female_edu_yrs", "gdp_k", "urban_pct",
             "contracep_pct", "infant_mort", "median_age"]

    print("=== Birth Rate Analysis System (from scratch) ===")
    print(f"test regions          : {len(yte)}")
    print(f"birth rate range      : {yte.min():.1f}-{yte.max():.1f} per 1000")
    print(f"baseline (mean-rate)  : RMSE={base_rmse:.3f}")
    print(f"ridge regression model: RMSE={model_rmse:.3f}  R^2={model_r2:.3f}")
    print(f"error reduction       : {(1 - model_rmse / base_rmse) * 100:.1f}% vs baseline")
    print("recovered effects (birth-rate change per unit):")
    for name, c in zip(names, model.coefficients()):
        print(f"  {name:<15}: {c:+.3f}")
    assert model_rmse < 0.5 * base_rmse, "model should clearly beat the mean baseline"
    print("PASS: birth-rate model beats the mean-rate baseline")
