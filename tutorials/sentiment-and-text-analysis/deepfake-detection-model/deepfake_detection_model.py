import numpy as np

# Deepfake TEXT detection: tell apart authentic human writing from machine
# "deepfake" text. Machine text leaves statistical fingerprints -- a narrower
# vocabulary, heavy word repetition, and generic filler transitions -- so we
# combine a bag-of-words view with hand-built stylometric features and train a
# from-scratch logistic-regression detector on them.

# ---- Word pools that plant the latent human-vs-machine signal. ----
HUMAN_WORDS = ["honestly", "gonna", "kinda", "weird", "loved", "hated", "messy",
               "coffee", "rain", "friend", "argued", "laughed", "cried", "cheap",
               "brilliant", "awful", "cat", "grandma", "tuesday", "sunburn",
               "detour", "leftovers", "mixtape", "sneakers", "gossip", "hiccup",
               "doodle", "midnight", "sarcasm", "crumbs", "whisper", "stubborn"]
FAKE_WORDS = ["furthermore", "additionally", "moreover", "essentially", "overall",
              "utilize", "leverage", "optimal", "comprehensive", "seamless",
              "robust", "holistic", "synergy", "framework", "solution",
              "innovative", "streamline", "paradigm", "facilitate", "delve"]
SHARED = ["the", "a", "and", "to", "of", "it", "was", "is", "this", "that",
          "with", "for", "very", "really", "so", "then", "day", "time"]


def tokenize(text):
    return text.lower().split()


def make_corpus(n=800, seed=0):
    """Synthetic labelled text. Human docs draw from a wide vocabulary with
    high lexical variety; deepfake docs lean on generic filler and repeat a
    few tokens, lowering diversity -- the fingerprint the detector recovers."""
    rng = np.random.RandomState(seed)
    docs, y = [], []
    for _ in range(n):
        if rng.rand() < 0.5:                        # label 0 = authentic human
            length = rng.randint(18, 30)
            words = list(rng.choice(HUMAN_WORDS, int(length * 0.6))) + \
                list(rng.choice(SHARED, length - int(length * 0.6)))
            y.append(0)
        else:                                        # label 1 = machine deepfake
            length = rng.randint(18, 30)
            n_fill = int(length * 0.45)
            base = list(rng.choice(FAKE_WORDS, n_fill)) + \
                list(rng.choice(SHARED, length - n_fill))
            # Machine text repeats itself: overwrite ~20% with one anchor word.
            anchor = rng.choice(FAKE_WORDS)
            for i in rng.choice(len(base), int(length * 0.2), replace=False):
                base[i] = anchor
            words = base
            y.append(1)
        rng.shuffle(words)
        docs.append(" ".join(words))
    return docs, np.array(y)


class CountVectorizer:
    """Bag-of-words: learn a vocabulary, then map docs to term-count rows."""

    def fit(self, docs):
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)              # skip unseen words
                if j is not None:
                    X[i, j] += 1.0
        return X


def stylometry(docs):
    """Hand-built forensic features that expose machine text regardless of
    exact words: lexical diversity, top-token repetition, mean word length."""
    feats = np.zeros((len(docs), 3))
    for i, d in enumerate(docs):
        toks = tokenize(d)
        counts = np.array([toks.count(w) for w in set(toks)])
        feats[i, 0] = len(set(toks)) / len(toks)           # type-token ratio
        feats[i, 1] = counts.max() / len(toks)             # peak repetition
        feats[i, 2] = np.mean([len(w) for w in toks])      # avg word length
    return feats


class LogisticRegression:
    """Binary logistic regression from scratch: full-batch gradient descent on
    the cross-entropy loss with L2 regularisation, standardised features."""

    def __init__(self, lr=0.5, epochs=400, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b)))
            g = p - y                                       # dL/dz
            self.w -= self.lr * (Xs.T @ g / n + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def predict_proba(self, X):
        Xs = (X - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b)))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def auc(y_true, scores):
    """ROC AUC via the Mann-Whitney rank statistic (no sklearn)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = y_true == 1, y_true == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def prf(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)

    docs, y = make_corpus(n=800, seed=0)

    # Held-out split: fit on 70%, evaluate on the untouched 30%.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    dtr, dte = [docs[i] for i in tr], [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = CountVectorizer().fit(dtr)
    # Feature vector = bag-of-words  ++  3 stylometric forensic features.
    Xtr = np.hstack([vec.transform(dtr), stylometry(dtr)])
    Xte = np.hstack([vec.transform(dte), stylometry(dte)])

    clf = LogisticRegression().fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    prec, rec, f1 = prf(yte, pred)
    model_auc = auc(yte, proba)

    # Baselines: majority-class label, and a random detector (AUC = 0.5).
    majority = np.bincount(ytr).argmax()
    base_acc = np.mean(yte == majority)

    print("Docs: %d   Train: %d   Test: %d   Vocab: %d + 3 stylometric feats"
          % (len(docs), len(tr), len(te), len(vec.vocab)))
    print("-" * 62)
    print("Deepfake detector  accuracy: %.4f  P: %.3f  R: %.3f  F1: %.3f"
          % (acc, prec, rec, f1))
    print("Detector ROC AUC : %.4f   (random detector = 0.5000)" % model_auc)
    print("Majority baseline accuracy: %.4f" % base_acc)
    print("-" * 62)
    for text in [" ".join(np.random.choice(HUMAN_WORDS, 12)),
                 " ".join(["synergy"] * 4 + list(np.random.choice(FAKE_WORDS, 8)))]:
        f = np.hstack([vec.transform([text]), stylometry([text])])
        p = clf.predict_proba(f)[0]
        print("  p(deepfake)=%.3f -> %-8s  '%s...'"
              % (p, "DEEPFAKE" if p >= 0.5 else "HUMAN", text[:34]))
    print("-" * 62)
    print("Beats baselines (acc>majority and AUC>0.5): %s"
          % (acc > base_acc and model_auc > 0.5))
