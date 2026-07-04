import numpy as np


class MatchPredictor:
    """Logistic-regression Tinder match predictor trained from scratch.

    A "match" is mutual interest: it happens when both people would swipe right.
    Given features describing a viewer/candidate pairing, the model standardizes
    them, learns weights + bias via full-batch gradient descent on the binary
    cross-entropy loss, and predicts the match probability / label.
    """

    def __init__(self, lr=0.2, n_iter=3000, l2=1e-3):
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
        # scaling stats from the training set only
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = self._sigmoid(Xs @ self.w + self.b)  # predicted match probs
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


def make_swipe_data(n=1200):
    """Synthetic swipe interactions with a planted mutual-attraction rule.

    Features per viewer/candidate pairing:
      age_gap        : |age difference| in years
      distance_km    : how far apart they live
      shared_interests: overlapping hobbies (0..6)
      attract_gap    : mismatch in attractiveness ratings
      bio_similarity : text/profile similarity (0..1)
      both_active    : both users open the app often (0/1)
    A match (mutual right-swipe) is likely when the gap features are small and the
    similarity/interest/activity features are high.
    """
    age_gap = np.abs(np.random.normal(0, 6, n))
    distance_km = np.abs(np.random.normal(0, 25, n))
    shared_interests = np.random.randint(0, 7, n)
    attract_gap = np.abs(np.random.normal(0, 2.0, n))
    bio_similarity = np.clip(np.random.beta(2, 2, n), 0, 1)
    both_active = (np.random.rand(n) > 0.5).astype(float)
    X = np.column_stack([age_gap, distance_km, shared_interests,
                         attract_gap, bio_similarity, both_active])

    # planted logit for a mutual match: penalties for gaps, rewards for overlap
    z = (-0.18 * age_gap - 0.05 * distance_km + 0.45 * shared_interests
         - 0.55 * attract_gap + 2.2 * bio_similarity + 0.8 * both_active - 1.4)
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (prob + np.random.normal(0, 0.12, n) > 0.5).astype(int)
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


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_swipe_data(1200)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = MatchPredictor(lr=0.2, n_iter=3000, l2=1e-3).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    pred = (proba >= 0.5).astype(int)

    acc = np.mean(pred == yte)
    majority = int(round(ytr.mean()))          # predict the most common class
    base_acc = np.mean(yte == majority)
    auc = roc_auc(yte, proba)

    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    feats = ["age_gap", "distance", "shared_int", "attract_gap", "bio_sim", "active"]
    print("test interactions     :", len(yte))
    print("match rate (test)     :", round(float(yte.mean()), 3))
    print("majority baseline acc :", round(float(base_acc), 3))
    print("model accuracy        :", round(float(acc), 3))
    print("precision / recall    :", round(float(precision), 3), "/", round(float(recall), 3))
    print("F1 score              :", round(float(f1), 3))
    print("ROC AUC (0.5=random)  :", round(float(auc), 3))
    print("learned weights       :", dict(zip(feats, np.round(model.w, 2))))
    print("BEATS baseline        :", bool(acc > base_acc and auc > 0.5))
