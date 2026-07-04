import numpy as np


def confusion_counts(y_true, y_pred, positive):
    """True/false positives and negatives for one class treated as 'positive'."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == positive) & (y_true == positive)))
    fp = int(np.sum((y_pred == positive) & (y_true != positive)))
    fn = int(np.sum((y_pred != positive) & (y_true == positive)))
    tn = int(np.sum((y_pred != positive) & (y_true != positive)))
    return tp, fp, fn, tn


def precision_recall_f1(y_true, y_pred, positive=1):
    """Binary/one-vs-rest precision, recall and F1 from raw counts.
    F1 = 2*P*R/(P+R) is the harmonic mean of precision and recall."""
    tp, fp, fn, _ = confusion_counts(y_true, y_pred, positive)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    return precision, recall, f1


def f1_score(y_true, y_pred, average="binary", positive=1):
    """F1-score supporting binary and multiclass averaging schemes:
    - binary:   F1 of the single 'positive' class.
    - macro:    unweighted mean of per-class F1.
    - weighted: per-class F1 weighted by class support (true count).
    - micro:    F1 from globally pooled tp/fp/fn (== accuracy for single-label)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == "binary":
        return precision_recall_f1(y_true, y_pred, positive)[2]

    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "micro":
        tp = fp = fn = 0
        for c in classes:
            a, b, d, _ = confusion_counts(y_true, y_pred, c)
            tp += a; fp += b; fn += d
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    f1s, supports = [], []
    for c in classes:
        f1s.append(precision_recall_f1(y_true, y_pred, c)[2])
        supports.append(int(np.sum(y_true == c)))
    f1s, supports = np.array(f1s), np.array(supports)

    if average == "macro":
        return float(np.mean(f1s))
    if average == "weighted":
        return float(np.sum(f1s * supports) / np.sum(supports))
    raise ValueError("unknown average: {}".format(average))


if __name__ == "__main__":
    np.random.seed(0)

    # HAND-VERIFIABLE CHECK: 8 samples with counts we can tally by eye.
    # positive=1. Pairs (true,pred): TP=3, FP=1, FN=2, TN=2.
    yt = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    yp = np.array([1, 1, 1, 0, 0, 1, 0, 0])
    p, r, f1 = precision_recall_f1(yt, yp, positive=1)
    # P = 3/4 = 0.75, R = 3/5 = 0.6, F1 = 2*.75*.6/1.35 = 0.6667.
    print("Hand check -> precision={:.4f} recall={:.4f} f1={:.4f}".format(p, r, f1))
    assert abs(p - 0.75) < 1e-9 and abs(r - 0.6) < 1e-9
    assert abs(f1 - 2 / 3) < 1e-9, "binary F1 formula is wrong"
    print("PASS: binary F1 matches the tallied-by-hand value 0.6667.")

    # PLANTED STRUCTURE: an IMBALANCED binary problem (10% positives) where a
    # linear signal separates the classes. F1 is the honest metric here because a
    # trivial always-negative model scores high accuracy but F1 = 0.
    n, d = 600, 4
    y = (np.random.rand(n) < 0.10).astype(int)           # rare positive class
    X = np.random.randn(n, d)
    w = np.array([1.6, -1.2, 0.9, 0.0])
    X[y == 1] += w                                        # shift positives along w
    score = X @ w
    thr = np.quantile(score, 0.90)                        # flag top 10% as positive
    pred = (score > thr).astype(int)

    # BASELINE: majority classifier always predicts the negative (majority) class.
    majority = np.zeros(n, dtype=int)

    acc_model = np.mean(pred == y)
    acc_major = np.mean(majority == y)
    f1_model = f1_score(y, pred, average="binary", positive=1)
    f1_major = f1_score(y, majority, average="binary", positive=1)

    print("\nImbalanced binary task (positives = {:.0%}):".format(y.mean()))
    print("Majority baseline: accuracy={:.3f}  F1={:.3f}".format(acc_major, f1_major))
    print("Signal model:      accuracy={:.3f}  F1={:.3f}".format(acc_model, f1_model))
    print("F1 improvement over majority baseline: +{:.3f}".format(f1_model - f1_major))

    # MULTICLASS DEMO: 3 planted clusters classified by nearest centroid; report
    # macro / micro / weighted F1 (all high because the clusters are separable).
    centers = np.array([[0, 0], [6, 6], [0, 6]], dtype=float)
    ym = np.repeat([0, 1, 2], 120)
    Xm = centers[ym] + np.random.randn(len(ym), 2) * 0.9
    dists = np.linalg.norm(Xm[:, None, :] - centers[None, :, :], axis=2)
    predm = np.argmin(dists, axis=1)
    print("\nMulticlass (3 clusters) nearest-centroid classifier:")
    print("Macro F1   : {:.3f}".format(f1_score(ym, predm, average="macro")))
    print("Micro F1   : {:.3f}".format(f1_score(ym, predm, average="micro")))
    print("Weighted F1: {:.3f}".format(f1_score(ym, predm, average="weighted")))

    assert f1_major == 0.0, "majority model must have F1 = 0 on the rare class"
    assert f1_model > 0.6, "signal model F1 should clearly beat the baseline"
    assert f1_score(ym, predm, average="macro") > 0.9, "multiclass F1 too low"
    print("\nPASS: F1 correctly rewards the real model and zeroes the majority baseline.")
