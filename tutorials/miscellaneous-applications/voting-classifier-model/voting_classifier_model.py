import numpy as np


# --- three diverse base classifiers, each built from scratch ---------------

class LogisticRegression:
    """Binary logistic regression via full-batch gradient descent."""

    def __init__(self, lr=0.1, n_iter=500):
        self.lr = lr
        self.n_iter = n_iter

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = (X - self.mu) / self.sigma
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = self._sigmoid(Xs @ self.w + self.b)
            err = p - y
            self.w -= self.lr * (Xs.T @ err / n)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        p1 = self._sigmoid(((X - self.mu) / self.sigma) @ self.w + self.b)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class GaussianNB:
    """Gaussian Naive Bayes: per-class feature means/variances."""

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.theta, self.var, self.prior = [], [], []
        for c in self.classes:
            Xc = X[y == c]
            self.theta.append(Xc.mean(axis=0))
            self.var.append(Xc.var(axis=0) + 1e-6)
            self.prior.append(len(Xc) / len(X))
        self.theta = np.array(self.theta)
        self.var = np.array(self.var)
        self.prior = np.array(self.prior)
        return self

    def predict_proba(self, X):
        log_post = []
        for i in range(len(self.classes)):
            ll = -0.5 * np.sum(np.log(2 * np.pi * self.var[i])
                               + (X - self.theta[i]) ** 2 / self.var[i], axis=1)
            log_post.append(ll + np.log(self.prior[i]))
        log_post = np.array(log_post).T
        log_post -= log_post.max(axis=1, keepdims=True)  # stability
        p = np.exp(log_post)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class KNN:
    """k-nearest-neighbours; proba = fraction of neighbours per class."""

    def __init__(self, k=15):
        self.k = k

    def fit(self, X, y):
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        self.X = (X - self.mu) / self.sigma
        self.y = y
        self.classes = np.unique(y)
        return self

    def predict_proba(self, X):
        Xs = (X - self.mu) / self.sigma
        # squared euclidean distances (test x train), vectorized
        d = (np.sum(Xs ** 2, axis=1)[:, None]
             + np.sum(self.X ** 2, axis=1)[None, :]
             - 2 * Xs @ self.X.T)
        nn = np.argsort(d, axis=1)[:, :self.k]           # nearest indices
        votes = self.y[nn]
        return np.column_stack([(votes == c).mean(axis=1) for c in self.classes])

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# --- the ensemble ----------------------------------------------------------

class VotingClassifier:
    """Combine specialist base models by hard (majority) or soft (avg proba) vote.

    Each estimator is a (name, model, cols) triple and is trained on its OWN
    feature view `cols`.  Because the specialists read disjoint signals, their
    errors decorrelate -- exactly the condition under which a vote beats every
    individual member.
    """

    def __init__(self, estimators, voting="soft"):
        self.estimators = estimators  # list of (name, model, cols)
        self.voting = voting

    def fit(self, X, y):
        self.classes = np.unique(y)
        for _, m, cols in self.estimators:
            m.fit(X[:, cols], y)
        return self

    def predict(self, X):
        if self.voting == "soft":
            # average the predicted class-probabilities across specialists
            avg = np.mean([m.predict_proba(X[:, c])
                           for _, m, c in self.estimators], axis=0)
            return self.classes[avg.argmax(axis=1)]
        # hard voting: majority predicted label across specialists
        preds = np.array([m.predict(X[:, c]) for _, m, c in self.estimators])
        return np.array([np.bincount(preds[:, i]).argmax()
                         for i in range(X.shape[0])])


# feature-view blocks: each base model only sees its own columns
COLS_LINEAR = [0, 1]      # linear mean-shift  -> logistic regression
COLS_VAR = [2, 3]         # variance-shift     -> Gaussian naive Bayes
COLS_XOR = [4, 5, 6]      # XOR gate signal    -> KNN


def make_data(n=1400):
    """Two classes encoded in three DISJOINT feature blocks of different type.

    - block A (a, b): a linear mean shift -> only logistic regression reads it
    - block B (v1, v2): equal means but class-dependent VARIANCE -> naive Bayes
    - block C (g1, g2, e): an XOR interaction -> only the distance-based KNN
    Each block alone is weak/noisy, so every specialist is mediocre on its own
    but the three make independent mistakes -> the vote recovers the full label.
    """
    y = np.random.randint(0, 2, n)
    # block A: linear separation with heavy Gaussian overlap
    a = np.random.normal(0, 2.0, n) + np.where(y == 1, 1.1, -1.1)
    b = np.random.normal(0, 2.0, n) + np.where(y == 1, -1.1, 1.1)
    # block B: same mean, class-dependent spread (invisible to linear models)
    v1 = np.random.normal(0, np.where(y == 1, 3.0, 1.0), n)
    v2 = np.random.normal(0, np.where(y == 1, 1.0, 3.0), n)
    # block C: two gates whose XOR encodes the class (naive/linear models blind)
    g1 = np.random.normal(0, 1.0, n)
    g2 = np.random.normal(0, 1.0, n)
    flip = ((g1 > 0) ^ (g2 > 0))
    e = np.where(flip == (y == 1), 1.0, -1.0) + np.random.normal(0, 0.8, n)
    return np.column_stack([a, b, v1, v2, g1, g2, e]), y


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_data(1400)
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    # three specialists, each on its own feature block (name, model, cols)
    members = [("logreg", LogisticRegression(), COLS_LINEAR),
               ("gnb", GaussianNB(), COLS_VAR),
               ("knn", KNN(k=15), COLS_XOR)]
    soft = VotingClassifier(members, voting="soft").fit(Xtr, ytr)
    hard = VotingClassifier(members, voting="hard")  # reuses fitted members

    # accuracy of each base specialist on the held-out set
    print("held-out test samples :", len(yte))
    accs = {}
    for name, m, c in members:
        accs[name] = float(np.mean(m.predict(Xte[:, c]) == yte))
        print("base %-7s acc     : %.3f" % (name, accs[name]))

    majority = np.bincount(ytr).argmax()
    base_acc = float(np.mean(yte == majority))
    best_single = max(accs.values())
    soft_acc = float(np.mean(soft.predict(Xte) == yte))
    hard_acc = float(np.mean(hard.predict(Xte) == yte))

    print("majority baseline acc :", round(base_acc, 3))
    print("best single model acc :", round(best_single, 3))
    print("HARD voting accuracy  :", round(hard_acc, 3))
    print("SOFT voting accuracy  :", round(soft_acc, 3))
    print("BEATS baseline        :", bool(soft_acc > base_acc))
    print("BEATS best single     :", bool(soft_acc >= best_single))
