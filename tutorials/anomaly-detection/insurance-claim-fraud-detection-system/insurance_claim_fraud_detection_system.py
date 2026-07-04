import numpy as np

# Insurance Claim Fraud Detection System (from scratch)
# -----------------------------------------------------
# Fraudulent claims are RARE and look "unusual" jointly across features: a big
# payout on a brand-new policy, reported late, by a customer with many prior
# claims and thin documentation. Each feature alone overlaps the honest crowd,
# so a per-feature z-score is weak -- the fraud only stands out in the JOINT
# feature space. We detect it with an ISOLATION FOREST built by hand: an
# ensemble of random binary trees that recursively cut the data on a random
# feature at a random threshold. Rare, out-of-pattern points get "isolated"
# (land in their own leaf) after FEW cuts, so a short average path length ->
# high anomaly score. We fit UNSUPERVISED on training claims, pick a threshold
# from an assumed contamination rate, then score a held-out LABELED test set.


def _c(n):
    """Expected path length of an unsuccessful BST search over n points --
    the normalization that turns raw depths into a comparable score."""
    if n > 2:
        return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n
    return 1.0 if n == 2 else 0.0


class IsolationForest:
    """Ensemble of random isolation trees; anomaly = short average path."""

    def __init__(self, n_trees=120, sample_size=256, contamination=0.05, seed=0):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.contamination = contamination
        self.rng = np.random.RandomState(seed)

    def _build(self, X, depth):
        # External (leaf) node once we hit the height limit or can't split.
        n = len(X)
        if depth >= self.height_limit or n <= 1:
            return {"size": n}
        # Pick a random feature that actually varies, then a random split point.
        feats = self.rng.permutation(X.shape[1])
        for q in feats:
            lo, hi = X[:, q].min(), X[:, q].max()
            if hi > lo:
                p = self.rng.uniform(lo, hi)
                mask = X[:, q] < p
                return {"q": q, "p": p,
                        "L": self._build(X[mask], depth + 1),
                        "R": self._build(X[~mask], depth + 1)}
        return {"size": n}                       # all features constant -> leaf

    def fit(self, X):
        n = len(X)
        self.sample_size = min(self.sample_size, n)
        self.height_limit = int(np.ceil(np.log2(self.sample_size)))
        self.trees = [self._build(X[self.rng.choice(n, self.sample_size, replace=False)], 0)
                      for _ in range(self.n_trees)]
        # Threshold = upper-contamination quantile of the training anomaly scores.
        self.threshold_ = np.quantile(self.score_samples(X), 1.0 - self.contamination)
        return self

    def _path(self, x, node, depth):
        # Depth reached when x lands in a leaf, plus the leaf's expected sub-depth.
        if "size" in node:
            return depth + _c(node["size"])
        branch = "L" if x[node["q"]] < node["p"] else "R"
        return self._path(x, node[branch], depth + 1)

    def score_samples(self, X):
        # Mean path over all trees -> normalized isolation score in (0, 1);
        # higher = isolated faster = more likely fraud.
        depths = np.array([[self._path(x, t, 0) for t in self.trees] for x in X])
        return 2.0 ** (-depths.mean(axis=1) / _c(self.sample_size))

    def predict(self, X):
        return (self.score_samples(X) > self.threshold_).astype(int)


# ------------------------------ metric helpers ------------------------------
def roc_auc(labels, scores):
    """Rank-based ROC AUC (Mann-Whitney): P(score[fraud] > score[honest])."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def prf(labels, preds):
    """Precision, recall, F1 for the positive (fraud) class."""
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def make_claims(n, fraud):
    """Synthetic insurance claims. Honest and fraudulent draws OVERLAP on every
    single feature but separate in the joint space -- exactly what isolation
    forest exploits. Columns: amount, report_delay, prior_claims, tenure_mo,
    claim/premium ratio, supporting_docs."""
    if fraud:
        cols = [np.random.normal(13, 3, n),      # bigger payout
                np.random.normal(9, 3, n),       # reported late
                np.random.poisson(3, n),         # many prior claims
                np.random.normal(20, 10, n),     # new customer
                np.random.normal(0.60, 0.15, n), # high claim/premium
                np.random.normal(3.5, 1.5, n)]   # thin documentation
    else:
        cols = [np.random.normal(8, 2.5, n),
                np.random.normal(4, 2, n),
                np.random.poisson(1, n),
                np.random.normal(48, 15, n),
                np.random.normal(0.35, 0.10, n),
                np.random.normal(6, 1.5, n)]
    return np.column_stack(cols)


if __name__ == "__main__":
    np.random.seed(0)
    n_honest, n_fraud = 950, 50               # 5% fraud prevalence

    X = np.vstack([make_claims(n_honest, False), make_claims(n_fraud, True)])
    y = np.r_[np.zeros(n_honest), np.ones(n_fraud)].astype(int)

    # Shuffle, then split into an UNLABELED train set and a LABELED test set.
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    n_tr = 600
    X_train, X_test, y_test = X[:n_tr], X[n_tr:], y[n_tr:]

    forest = IsolationForest(n_trees=120, sample_size=256, contamination=0.05).fit(X_train)
    scores = forest.score_samples(X_test)
    preds = forest.predict(X_test)

    prec, rec, f1 = prf(y_test, preds)
    auc = roc_auc(y_test, scores)

    # Baselines: random ranking (AUC 0.5) and majority "flag nothing".
    prevalence = y_test.mean()
    _, _, maj_f1 = prf(y_test, np.zeros_like(y_test))

    print("Insurance Claim Fraud Detection via from-scratch Isolation Forest")
    print("-" * 66)
    print("features=%d  train(unlabeled)=%d  test=%d  fraud=%d (%.1f%%)"
          % (X.shape[1], n_tr, len(y_test), int(y_test.sum()), 100 * prevalence))
    print("trees=%d  subsample=%d  isolation-score threshold=%.3f"
          % (forest.n_trees, forest.sample_size, forest.threshold_))
    print("-" * 66)
    print("IsolationForest  AUC=%.3f  Precision=%.3f  Recall=%.3f  F1=%.3f"
          % (auc, prec, rec, f1))
    print("Random rank      AUC=0.500")
    print("Majority(honest) Precision=0.000  Recall=0.000  F1=%.3f" % maj_f1)
    print("-" * 66)
    print("AUC lift over random:   0.500 -> %.3f" % auc)
    print("F1  lift over majority: %.3f -> %.3f" % (maj_f1, f1))
    ok = auc > 0.85 and f1 > 0.5 and f1 > maj_f1
    print("RESULT:", "PASS - clearly beats baselines" if ok else "FAIL")
