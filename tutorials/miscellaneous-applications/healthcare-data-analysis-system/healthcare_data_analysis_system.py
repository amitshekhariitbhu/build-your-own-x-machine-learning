import numpy as np


class HealthcareRiskAnalyzer:
    """Logistic-regression clinical risk model from scratch.

    Screens patients as high-risk / low-risk for a chronic condition from
    routine vitals and labs. Trained by full-batch gradient descent on the
    binary cross-entropy loss over standardized features; every step of the
    math (sigmoid, gradient, update) is manual numpy, no ML library.
    """

    def __init__(self, lr=0.3, epochs=400, l2=1e-3):
        self.lr = lr            # gradient-descent step size
        self.epochs = epochs    # number of full passes
        self.l2 = l2            # ridge penalty on weights
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    def _standardize(self, X, fit=False):
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0) + 1e-8
        return (X - self.mu) / self.sigma

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = self._standardize(np.asarray(X, dtype=float), fit=True)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)   # predicted risk
            err = p - y                              # BCE gradient wrt logit
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        X = self._standardize(np.asarray(X, dtype=float))
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def odds_ratios(self):
        """exp(weight) per standardized feature: >1 raises risk, <1 lowers it."""
        return np.exp(self.w)


def make_patient_data(n=900, seed=0):
    """Synthetic patient records with a planted disease-risk signal.

    Feature order:
      age            - years                       (up   -> more risk)
      bmi            - body-mass index              (up   -> more risk)
      systolic_bp    - systolic blood pressure mmHg (up   -> more risk)
      glucose        - fasting glucose mg/dL        (up   -> more risk)
      cholesterol    - total cholesterol mg/dL      (up   -> more risk)
      resting_hr     - resting heart rate bpm       (up   -> more risk)
      exercise_hrs   - weekly exercise hours        (up   -> less risk)
    """
    rng = np.random.RandomState(seed)
    age = rng.uniform(25, 80, n)
    bmi = rng.uniform(18, 42, n)
    systolic = rng.uniform(95, 185, n)
    glucose = rng.uniform(70, 200, n)
    chol = rng.uniform(120, 300, n)
    hr = rng.uniform(50, 100, n)
    exercise = rng.uniform(0, 12, n)
    X = np.column_stack([age, bmi, systolic, glucose, chol, hr, exercise])

    # planted log-odds generating process on standardized-ish scaled features
    z = (0.045 * (age - 52) + 0.10 * (bmi - 28) + 0.035 * (systolic - 130)
         + 0.030 * (glucose - 110) + 0.012 * (chol - 200) + 0.03 * (hr - 72)
         - 0.22 * (exercise - 5) - 0.3)
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(0, 1, n) < prob).astype(int)  # Bernoulli labels
    return X, y


def confusion(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn


def metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion(y_true, y_pred)
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_patient_data(n=900, seed=0)

    # held-out split: 70% train / 30% test
    n_train = int(0.7 * len(y))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    model = HealthcareRiskAnalyzer(lr=0.3, epochs=400).fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc, prec, rec, f1 = metrics(yte, pred)

    # baseline: always predict the majority class from the training set
    majority = int(round(ytr.mean()))
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)

    names = ["age", "bmi", "systolic_bp", "glucose",
             "cholesterol", "resting_hr", "exercise_hrs"]

    print("=== Healthcare Data Analysis System (from scratch) ===")
    print(f"test patients          : {len(yte)}")
    print(f"high-risk prevalence   : {yte.mean() * 100:.1f}%")
    print(f"baseline (majority)    : accuracy={base_acc:.3f}")
    print(f"logistic risk model    : accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    print(f"accuracy lift          : {(acc - base_acc) * 100:+.1f} points vs majority")
    print("recovered risk factors (odds ratio per std-unit):")
    for name, o in zip(names, model.odds_ratios()):
        arrow = "raises" if o > 1 else "lowers"
        print(f"  {name:<13}: {o:.2f}x ({arrow} risk)")
    assert acc > base_acc + 0.08, "model should clearly beat the majority baseline"
    print("PASS: risk model beats the majority-class baseline")
