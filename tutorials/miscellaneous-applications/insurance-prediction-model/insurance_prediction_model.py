import numpy as np


class InsuranceClaimPredictor:
    """Logistic-regression claim predictor trained fully from scratch.

    Insurers price policies on the RISK that a holder files a claim next term.
    This model mixes CATEGORICAL context (vehicle type, region) with NUMERIC
    risk signals (age, driving experience, past claims, annual mileage). It
    one-hot encodes categoricals, standardizes numerics, and learns weights +
    bias by full-batch gradient descent on the binary cross-entropy objective.
    No ML libraries -- just numpy math.
    """

    def __init__(self, lr=0.3, n_iter=4000, l2=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # L2 regularization strength
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _standardize(self, X):
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        # scaling stats learned from the training set only (avoids leakage)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = self._sigmoid(Xs @ self.w + self.b)  # predicted claim probs
            err = p - y
            grad_w = Xs.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        Xs = self._standardize(np.asarray(X, dtype=float))
        return self._sigmoid(Xs @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def _one_hot(idx, k):
    """Encode integer category codes (0..k-1) as one-hot columns."""
    oh = np.zeros((len(idx), k))
    oh[np.arange(len(idx)), idx] = 1.0
    return oh


def make_insurance_data(n=4000):
    """Synthetic auto-insurance policy holders with a planted claim rule.

    Raw fields per policy:
      vehicle   : 0=sedan 1=suv 2=sports      (sports cars crash far more)
      region    : 0=rural 1=suburban 2=urban  (urban traffic raises risk)
      age       : driver age in years         (young drivers are riskier)
      experience: years holding a license
      past_claims: claims filed in prior years
      mileage   : annual km driven (thousands)
    Whether a holder files a claim depends mostly on youth, past claims,
    vehicle type and mileage -- the latent risk signal the model must recover.
    """
    vehicle = np.random.randint(0, 3, n)
    region = np.random.randint(0, 3, n)
    age = np.random.randint(18, 75, n)
    experience = np.clip(age - 18 - np.random.randint(0, 5, n), 0, None)
    past_claims = np.random.poisson(0.6, n)
    mileage = np.abs(np.random.normal(15, 6, n))

    # planted claim logit: risk falls with age/experience, rises with prior
    # claims, mileage, sporty cars and dense urban regions.
    veh_effect = np.array([-0.2, 0.1, 1.3])[vehicle]     # sports >> suv > sedan
    reg_effect = np.array([-0.4, 0.0, 0.6])[region]      # urban riskiest
    z = (veh_effect + reg_effect
         - 0.045 * (age - 30) - 0.05 * experience
         + 0.9 * past_claims + 0.05 * (mileage - 15) - 1.1)
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (np.random.rand(n) < prob).astype(int)  # sample real Bernoulli claims

    # model feature matrix: one-hot categoricals + numeric risk columns
    X = np.column_stack([
        _one_hot(vehicle, 3), _one_hot(region, 3),
        age, experience, past_claims, mileage,
    ])
    return X, y


def roc_auc(y_true, scores):
    """AUC via the rank-sum (Mann-Whitney U) identity, computed from scratch."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)  # 1-based ranks
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def log_loss(y_true, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_insurance_data(4000)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = InsuranceClaimPredictor(lr=0.3, n_iter=4000, l2=1e-3).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    pred = (proba >= 0.5).astype(int)

    acc = np.mean(pred == yte)
    majority = int(round(ytr.mean()))                 # predict most common class
    base_acc = np.mean(yte == majority)
    base_rate = ytr.mean()                            # constant claim-rate baseline
    auc = roc_auc(yte, proba)
    ll_model = log_loss(yte, proba)
    ll_base = log_loss(yte, np.full(len(yte), base_rate))

    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("test policies          :", len(yte))
    print("claim rate (test)      :", round(float(yte.mean()), 3))
    print("majority baseline acc  :", round(float(base_acc), 3))
    print("model accuracy         :", round(float(acc), 3))
    print("precision / recall     :", round(float(precision), 3), "/", round(float(recall), 3))
    print("F1 score               :", round(float(f1), 3))
    print("ROC AUC  (0.5=random)  :", round(float(auc), 3))
    print("log-loss constant-rate :", round(float(ll_base), 4))
    print("log-loss model (lower) :", round(float(ll_model), 4))
    print("BEATS baseline         :", bool(acc > base_acc and auc > 0.5 and ll_model < ll_base))
