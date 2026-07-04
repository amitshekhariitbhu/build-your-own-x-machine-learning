import numpy as np


class LifeExpectancyRegressor:
    """Ridge-regularized linear regression from scratch (closed-form normal eq).

    Predicts a country-year's life expectancy (years) from health and
    socioeconomic indicators. Features are standardized; the ridge penalty keeps
    weights stable under correlated predictors. Solved with the normal equations
    W = (X'X + lambda*I)^-1 X'y so training is a single linear-algebra step.
    """

    def __init__(self, l2=1.0):
        self.l2 = l2          # ridge strength on standardized features
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
        self.b = y.mean()                       # intercept = target mean
        yc = y - self.b
        n, d = X.shape
        A = X.T @ X + self.l2 * np.eye(d)        # ridge-regularized Gram matrix
        self.w = np.linalg.solve(A, X.T @ yc)    # normal equations
        return self

    def predict(self, X):
        X = self._standardize(np.asarray(X, dtype=float))
        return X @ self.w + self.b


def make_life_expectancy_data(n=700, seed=0):
    """Synthetic country-year records with a planted life-expectancy signal.

    Feature order:
      adult_mortality  - deaths per 1000 adults (STRONGLY lowers life expectancy)
      infant_deaths    - infant deaths per 1000 (lowers)
      gdp_log          - log GDP per capita (raises)
      schooling        - avg years of schooling (raises)
      immunization     - vaccination coverage %  (raises)
      bmi              - population mean BMI (mild inverted effect via square)
      alcohol          - litres pure alcohol / capita / yr (mild lower)
      hiv_prevalence   - HIV/AIDS prevalence %  (STRONGLY lowers)
    Target = life expectancy in years, a nonlinear-ish function + noise.
    """
    rng = np.random.RandomState(seed)
    adult_mortality = rng.uniform(50, 400, n)
    infant_deaths = rng.uniform(0, 90, n)
    gdp_log = rng.uniform(5.5, 11.5, n)                 # ~ $250 .. $100k
    schooling = rng.uniform(3, 20, n)
    immunization = rng.uniform(40, 99, n)
    bmi = rng.uniform(18, 34, n)
    alcohol = rng.uniform(0, 16, n)
    hiv = rng.uniform(0.1, 25, n)

    # planted ground-truth relationship (coeffs in years-per-unit)
    life = (
        72.0
        - 0.045 * adult_mortality
        - 0.060 * infant_deaths
        + 1.60 * gdp_log
        + 0.75 * schooling
        + 0.10 * immunization
        - 0.020 * (bmi - 24.0) ** 2       # mild nonlinearity: extremes hurt
        - 0.25 * alcohol
        - 0.55 * hiv
    )
    life += rng.normal(0, 2.0, n)                        # observation noise
    life = np.clip(life, 40, 90)

    X = np.column_stack([adult_mortality, infant_deaths, gdp_log, schooling,
                         immunization, bmi, alcohol, hiv])
    return X, life


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_life_expectancy_data(n=700, seed=0)

    # held-out split: 70% train / 30% test
    n_train = int(0.7 * len(y))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    model = LifeExpectancyRegressor(l2=1.0).fit(Xtr, ytr)
    pred = model.predict(Xte)

    # baseline: always predict the mean life expectancy from the training set
    base_pred = np.full_like(yte, ytr.mean())

    m_rmse, m_mae, m_r2 = rmse(yte, pred), mae(yte, pred), r2_score(yte, pred)
    b_rmse, b_mae = rmse(yte, base_pred), mae(yte, base_pred)

    names = ["adult_mortality", "infant_deaths", "gdp_log", "schooling",
             "immunization", "bmi", "alcohol", "hiv"]
    order = np.argsort(-np.abs(model.w))

    print("=== Life Expectancy Analysis System (from scratch) ===")
    print(f"train / test records     : {len(ytr)} / {len(yte)}")
    print(f"life expectancy range    : {y.min():.1f} .. {y.max():.1f} years")
    print(f"baseline (predict mean)  : RMSE={b_rmse:.3f}  MAE={b_mae:.3f}")
    print(f"ridge regression         : RMSE={m_rmse:.3f}  MAE={m_mae:.3f}  R2={m_r2:.3f}")
    print(f"RMSE reduction vs mean   : {(1 - m_rmse / b_rmse) * 100:.1f}%")
    print("top drivers (|std weight|):")
    for i in order[:4]:
        sign = "+" if model.w[i] >= 0 else "-"
        print(f"    {names[i]:<16} {sign}{abs(model.w[i]):.2f}")
    assert m_rmse < 0.5 * b_rmse, "model should crush the mean baseline"
    print("PASS: ridge model beats mean baseline by a wide margin")
