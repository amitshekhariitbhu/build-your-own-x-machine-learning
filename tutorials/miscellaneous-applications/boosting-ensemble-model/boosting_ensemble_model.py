import numpy as np


class DecisionStump:
    """A one-split decision tree (weak learner): threshold on one feature.

    Chooses the (feature, threshold, polarity) that minimizes the weighted
    classification error over the sample weights supplied by the booster.
    Predicts labels in {-1, +1}.
    """

    def __init__(self):
        self.feature = 0
        self.threshold = 0.0
        self.polarity = 1  # +1: predict +1 when x >= thr, -1: flip

    def fit(self, X, y, w):
        n, d = X.shape
        best_err = np.inf
        for f in range(d):
            # candidate thresholds = midpoints between sorted unique values
            vals = np.unique(X[:, f])
            thresholds = (vals[:-1] + vals[1:]) / 2.0 if vals.size > 1 else vals
            for thr in thresholds:
                for pol in (1, -1):
                    pred = np.where(pol * X[:, f] >= pol * thr, 1, -1)
                    err = np.sum(w[pred != y])  # weighted error
                    if err < best_err:
                        best_err = err
                        self.feature, self.threshold, self.polarity = f, thr, pol
        return best_err

    def predict(self, X):
        p = self.polarity
        return np.where(p * X[:, self.feature] >= p * self.threshold, 1, -1)


class AdaBoost:
    """AdaBoost.M1 built from scratch on decision-stump weak learners.

    Each round: train a stump on the current sample weights, give it a vote
    alpha = 0.5 * ln((1 - err) / err), then up-weight the misclassified
    points so the next stump focuses on them. The final prediction is the
    sign of the alpha-weighted vote of all stumps.
    """

    def __init__(self, n_rounds=40):
        self.n_rounds = n_rounds
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)  # labels in {-1, +1}
        n = len(y)
        w = np.full(n, 1.0 / n)  # uniform sample weights
        self.stumps, self.alphas = [], []
        for _ in range(self.n_rounds):
            stump = DecisionStump()
            err = stump.fit(X, y, w)
            err = np.clip(err, 1e-10, 1 - 1e-10)
            if err >= 0.5:  # weak learner no better than chance: stop
                break
            alpha = 0.5 * np.log((1 - err) / err)  # stump vote weight
            pred = stump.predict(X)
            w *= np.exp(-alpha * y * pred)  # up-weight the misclassified
            w /= w.sum()  # renormalize to a distribution
            self.stumps.append(stump)
            self.alphas.append(alpha)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        agg = np.zeros(len(X))
        for alpha, stump in zip(self.alphas, self.stumps):
            agg += alpha * stump.predict(X)
        return agg

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


def make_data(n=800):
    """Synthetic 2-class data with a planted NON-LINEAR (XOR-like) boundary.

    Class is decided by the sign of x0 * x1 plus a soft radial term, so no
    single stump (and no linear model) can solve it, but a boosted ensemble
    of stumps can carve out the four quadrants. Label noise is added so the
    problem is not perfectly separable.
    """
    X = np.random.normal(0, 1, (n, 5))       # 2 signal dims + 3 noise dims
    signal = X[:, 0] * X[:, 1] + 0.6 * (X[:, 0] ** 2 - X[:, 1] ** 2)
    prob = 1.0 / (1.0 + np.exp(-3.0 * signal))
    y = np.where(prob + np.random.normal(0, 0.12, n) > 0.5, 1, -1)
    return X, y


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_data(800)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = AdaBoost(n_rounds=40).fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = np.mean(pred == yte)

    # baseline 1: majority class
    majority = 1 if ytr.mean() >= 0 else -1
    base_acc = np.mean(yte == majority)

    # baseline 2: a single decision stump (one weak learner, no boosting)
    stump = DecisionStump()
    stump.fit(Xtr, ytr, np.full(len(ytr), 1.0 / len(ytr)))
    stump_acc = np.mean(stump.predict(Xte) == yte)

    # precision / recall / F1 on the positive class
    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == -1))
    fn = np.sum((pred == -1) & (yte == 1))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("test samples          :", len(yte))
    print("rounds (stumps kept)  :", len(model.stumps))
    print("majority baseline acc :", round(float(base_acc), 3))
    print("single-stump acc      :", round(float(stump_acc), 3))
    print("AdaBoost accuracy     :", round(float(acc), 3))
    print("precision / recall    :", round(float(precision), 3), "/", round(float(recall), 3))
    print("F1 score              :", round(float(f1), 3))
    print("BEATS majority        :", bool(acc > base_acc))
    print("BEATS single stump    :", bool(acc > stump_acc))
