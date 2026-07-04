import numpy as np


def k_fold_indices(n, k, shuffle=True, seed=0):
    """Partition indices 0..n-1 into k disjoint test folds (sizes differ by <=1)."""
    idx = np.arange(n)
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    # np.array_split makes the folds as equal-sized as possible.
    return [np.sort(f) for f in np.array_split(idx, k)]


def stratified_k_fold_indices(y, k, shuffle=True, seed=0):
    """Partition indices into k folds while preserving each class's proportion per fold."""
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        if shuffle:
            rng.shuffle(c_idx)
        # Deal each class's samples across folds so per-fold class counts stay balanced.
        for j, chunk in enumerate(np.array_split(c_idx, k)):
            folds[j].extend(chunk.tolist())
    return [np.sort(np.array(f, dtype=int)) for f in folds]


def cross_val_score(make_model, X, y, folds):
    """Train a fresh model on the complement of each fold, score accuracy on that fold."""
    n = len(y)
    all_idx = np.arange(n)
    scores = []
    for test_idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False           # train on everything NOT in this fold
        model = make_model()
        model.fit(X[all_idx[mask]], y[all_idx[mask]])
        pred = model.predict(X[test_idx])
        scores.append(float(np.mean(pred == y[test_idx])))
    return np.array(scores)


class LogisticRegression:
    """Plain-numpy logistic regression trained by batch gradient descent."""

    def __init__(self, lr=0.3, epochs=1500):
        self.lr, self.epochs = lr, epochs

    def _design(self, X):
        return np.hstack([np.ones((len(X), 1)), X])   # prepend bias column

    def fit(self, X, y):
        Xb = self._design(X)
        self.w = np.zeros(Xb.shape[1])
        for _ in range(self.epochs):
            z = np.clip(Xb @ self.w, -30, 30)         # clip to avoid exp overflow
            p = 1.0 / (1.0 + np.exp(-z))
            self.w -= self.lr * (Xb.T @ (p - y) / len(y))
        return self

    def predict(self, X):
        z = np.clip(self._design(X) @ self.w, -30, 30)
        return (1.0 / (1.0 + np.exp(-z)) >= 0.5).astype(int)


if __name__ == "__main__":
    np.random.seed(0)

    # PLANTED STRUCTURE: 2 informative Gaussian-blob features (the real signal) plus
    # 40 pure-noise features. The ~30/70 class imbalance makes stratification matter,
    # and the noise features let an unregularized model overfit the small training pool.
    NOISE = 40
    def make_blobs(n0, n1):
        c0 = np.random.randn(n0, 2) * 1.1 + np.array([-1.4, -1.4])
        c1 = np.random.randn(n1, 2) * 1.1 + np.array([1.4, 1.4])
        X = np.vstack([c0, c1])
        X = np.hstack([X, np.random.randn(len(X), NOISE)])   # distractor dimensions
        y = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)
        perm = np.random.permutation(len(y))
        return X[perm], y[perm]

    # Small training pool for CV, plus a large held-out set = ground-truth generalization.
    X, y = make_blobs(140, 60)          # ~30% positives
    X_big, y_big = make_blobs(1400, 600)

    K = 5
    plain = k_fold_indices(len(y), K, seed=0)
    strat = stratified_k_fold_indices(y, K, seed=0)

    # --- Exact structural checks (hand-verifiable partition properties) ---
    covered = np.sort(np.concatenate(plain))
    assert np.array_equal(covered, np.arange(len(y))), "folds must cover every index once"
    assert sum(len(f) for f in plain) == len(y), "test folds must be disjoint & complete"
    assert max(len(f) for f in plain) - min(len(f) for f in plain) <= 1, "folds unbalanced"

    # Stratified folds must mirror the global positive rate closely; plain folds drift more.
    p_global = y.mean()
    strat_dev = max(abs(y[f].mean() - p_global) for f in strat)
    plain_dev = max(abs(y[f].mean() - p_global) for f in plain)

    make_model = lambda: LogisticRegression()

    # CV accuracy estimate vs. an optimistically-biased train-on-all / score-on-all number.
    cv_scores = cross_val_score(make_model, X, y, strat)
    cv_mean = cv_scores.mean()
    full = make_model().fit(X, y)
    train_acc = float(np.mean(full.predict(X) == y))
    true_acc = float(np.mean(full.predict(X_big) == y_big))   # ground-truth generalization

    majority = max(1 - p_global, p_global)                    # predict-the-majority baseline

    print("Class balance (train pool): {:.0%} pos / {:.0%} neg".format(p_global, 1 - p_global))
    print("Plain K-fold  test-fold sizes: {}".format([len(f) for f in plain]))
    print("Stratified    test-fold sizes: {}".format([len(f) for f in strat]))
    print("Max fold pos-rate deviation | plain: {:.3f}  stratified: {:.3f}".format(plain_dev, strat_dev))
    print("-" * 56)
    print("Majority-class baseline accuracy : {:.3f}".format(majority))
    print("Per-fold CV accuracies           : {}".format(np.round(cv_scores, 3).tolist()))
    print("Cross-validated accuracy (mean)  : {:.3f}".format(cv_mean))
    print("True held-out accuracy           : {:.3f}".format(true_acc))
    print("Optimistic train-on-all accuracy : {:.3f}".format(train_acc))
    print("|CV - true|  = {:.3f}   (CV is nearly unbiased)".format(abs(cv_mean - true_acc)))
    print("|train- true|= {:.3f}   (train accuracy is over-optimistic)".format(abs(train_acc - true_acc)))

    assert strat_dev <= plain_dev, "stratification should preserve class balance at least as well"
    assert cv_mean > majority + 0.15, "CV accuracy must clearly beat the majority baseline"
    assert abs(cv_mean - true_acc) < abs(train_acc - true_acc), "CV must estimate generalization better than train acc"
    print("PASS: valid partition, stratification preserved balance, CV beat baseline and"
          " estimated true accuracy better than the optimistic train score.")
