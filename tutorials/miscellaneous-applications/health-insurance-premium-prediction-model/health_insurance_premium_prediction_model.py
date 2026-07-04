import numpy as np


class PremiumRegressor:
    """Linear-regression insurance-premium predictor trained from scratch.

    Standardizes the input features, learns weights + bias by full-batch
    gradient descent on the L2-regularized mean-squared-error loss, and
    predicts the annual premium. No ML libraries -- just numpy math.
    """

    def __init__(self, lr=0.1, n_iter=4000, l2=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # L2 regularization strength
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    def _standardize(self, X):
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        # scaling stats computed on training data only (no test leakage)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = float(y.mean())  # start bias at the mean premium
        for _ in range(self.n_iter):
            pred = Xs @ self.w + self.b
            err = pred - y
            grad_w = Xs.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, X):
        Xs = self._standardize(np.asarray(X, dtype=float))
        return Xs @ self.w + self.b


def make_insurance_data(n=1000):
    """Synthetic policy-holder records with a planted premium signal.

    Features (insurance inspired):
      age, bmi, children, smoker, sex, region_risk.
    A true linear rule sets the premium (smoking dominates, age and bmi add
    cost), plus multiplicative + additive noise so it is not perfectly fit.
    """
    age = np.random.uniform(18, 64, n)
    bmi = np.random.normal(30, 6, n)
    children = np.random.randint(0, 6, n)
    smoker = np.random.binomial(1, 0.20, n)              # 1 = smoker
    sex = np.random.binomial(1, 0.50, n)                 # 1 = male
    region_risk = np.random.uniform(0, 1, n)             # regional cost index
    X = np.column_stack([age, bmi, children, smoker, sex, region_risk])

    # planted premium ($): smoking is by far the biggest driver, high bmi
    # amplifies the smoker penalty, age and children add steady cost.
    premium = (2500.0
               + 260.0 * age
               + 330.0 * bmi
               + 480.0 * children
               + 12000.0 * smoker
               + 900.0 * smoker * (bmi > 30)
               + 600.0 * sex
               + 3500.0 * region_risk)
    premium *= np.random.normal(1.0, 0.05, n)            # +-5% multiplicative
    premium += np.random.normal(0, 800, n)              # additive noise
    return X, premium


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_insurance_data(1000)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = PremiumRegressor(lr=0.1, n_iter=4000, l2=1e-3).fit(Xtr, ytr)
    pred = model.predict(Xte)

    # model error
    rmse = np.sqrt(np.mean((pred - yte) ** 2))
    mae = np.mean(np.abs(pred - yte))

    # mean-premium baseline (predict the training mean for everyone)
    base_pred = ytr.mean()
    base_rmse = np.sqrt(np.mean((base_pred - yte) ** 2))
    base_mae = np.mean(np.abs(base_pred - yte))

    # R^2 (fraction of premium variance explained on held-out data)
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    feats = ["age", "bmi", "children", "smoker", "sex", "region_risk"]

    print("test samples          :", len(yte))
    print("mean premium (test)   :", round(float(yte.mean()), 1))
    print("baseline RMSE (mean)  :", round(float(base_rmse), 1))
    print("model RMSE            :", round(float(rmse), 1))
    print("baseline MAE (mean)   :", round(float(base_mae), 1))
    print("model MAE             :", round(float(mae), 1))
    print("R^2 (held-out)        :", round(float(r2), 3))
    print("feature weights (std) :", dict(zip(feats, np.round(model.w, 1))))
    print("BEATS baseline        :", bool(rmse < base_rmse))
