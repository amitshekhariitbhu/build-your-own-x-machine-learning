import numpy as np


class ElectionPredictor:
    """Logistic-regression district-winner classifier from scratch.

    Each row is one electoral district described by campaign/demographic
    features. The model predicts whether PARTY A wins the district (1) or
    PARTY B wins (0). Trained by full-batch gradient descent on standardized
    features with an L2 penalty; outputs a win probability via the sigmoid,
    thresholded at 0.5. Summing predicted wins forecasts the national seat total.
    """

    def __init__(self, lr=0.3, n_iter=1500, l2=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # ridge penalty to keep weights sane
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def _standardize(self, X, fit=False):
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0) + 1e-8
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = self._standardize(np.asarray(X, dtype=float), fit=True)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = self._sigmoid(X @ self.w + self.b)  # predicted win probs
            err = p - y
            grad_w = X.T @ err / n + self.l2 * self.w  # cross-entropy + ridge grad
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        X = self._standardize(np.asarray(X, dtype=float))
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def make_election_data(n=1000, seed=0):
    """Synthetic districts with a planted, recoverable party-A win signal.

    Feature order:
      poll_margin    - poll lead for A minus B (points); strongest predictor
      incumbency     - +1 A holds seat, -1 B holds seat, 0 open
      econ_sentiment - economic/approval index favoring the incumbent side
      prev_margin    - A's margin in the previous election (points)
      ground_game    - A's campaign-spending advantage (normalized)
      partisan_lean  - district's baseline partisan lean toward A
    Labels are sampled from a logistic model of these features, so the true
    decision boundary exists but is noisy (competitive races flip either way).
    """
    rng = np.random.RandomState(seed)
    poll_margin = rng.normal(0.0, 6.0, n)
    incumbency = rng.choice([-1, 0, 1], size=n, p=[0.4, 0.2, 0.4])
    econ_sentiment = rng.normal(0.0, 1.0, n)
    prev_margin = rng.normal(0.0, 8.0, n)
    ground_game = rng.normal(0.0, 1.0, n)
    partisan_lean = rng.normal(0.0, 5.0, n)
    X = np.column_stack([poll_margin, incumbency, econ_sentiment,
                         prev_margin, ground_game, partisan_lean])

    # True generative weights: polls and prior/lean drive outcomes most.
    true_w = np.array([0.28, 0.55, 0.30, 0.11, 0.35, 0.09])
    logit = X @ true_w + rng.normal(0.0, 0.8, n)   # unmodeled campaign noise
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(int)   # A wins the district?
    return X, y


def metrics(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_election_data(n=1000, seed=0)

    # held-out split: 70% train / 30% test (the "election-day" districts)
    n_train = int(0.7 * len(y))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    clf = ElectionPredictor(lr=0.3, n_iter=1500, l2=1e-3).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc, prec, rec, f1 = metrics(yte, pred)

    # majority-class baseline (always call the more common training winner)
    majority = int(round(ytr.mean()))
    base_pred = np.full_like(yte, majority)
    base_acc, _, _, base_f1 = metrics(yte, base_pred)

    # national seat forecast: sum predicted A-wins across the test districts
    actual_seats = int(yte.sum())
    model_seats = int(pred.sum())
    base_seats = int(base_pred.sum())

    print("=== Election Prediction Model (from scratch) ===")
    print(f"test districts        : {len(yte)}")
    print(f"party-A wins (actual) : {actual_seats} / {len(yte)}")
    print(f"baseline (majority)   : acc={base_acc:.3f}  f1={base_f1:.3f}")
    print(f"logistic win model    : acc={acc:.3f}  f1={f1:.3f}")
    print(f"  precision={prec:.3f}  recall={rec:.3f}")
    print(f"improvement over base : +{(acc - base_acc) * 100:.1f} acc points")
    print(f"seat forecast  actual={actual_seats}  model={model_seats} "
          f"(err {abs(model_seats - actual_seats)})  "
          f"baseline={base_seats} (err {abs(base_seats - actual_seats)})")
    assert acc > base_acc + 0.10, "model should clearly beat the majority baseline"
    print("PASS: election model beats majority baseline")
