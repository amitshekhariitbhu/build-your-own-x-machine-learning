import numpy as np


def tokenize(text):
    return text.lower().split()


class BowVectorizer:
    """Bag-of-words with a shared vocabulary. Exposes two views of a corpus:
    raw counts (for Naive Bayes) and L2-normalized TF-IDF (for LogReg / kNN).
    idf = log((1 + N) / (1 + df)) + 1   (smoothed, sklearn-style).
    """

    def fit(self, docs):
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        df = np.zeros(len(self.vocab))
        for d in docs:
            for w in set(tokenize(d)):
                df[self.vocab[w]] += 1.0
        self.idf = np.log((1.0 + len(docs)) / (1.0 + df)) + 1.0
        return self

    def counts(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)            # ignore unseen words
                if j is not None:
                    X[i, j] += 1.0
        return X

    def tfidf(self, docs):
        X = self.counts(docs) * self.idf
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm == 0, 1.0, norm)


class MultinomialNB:
    """Multinomial Naive Bayes with Laplace smoothing on word counts."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.log_prior = np.zeros(len(self.classes))
        self.log_lik = np.zeros((len(self.classes), X.shape[1]))
        for k, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[k] = np.log(len(Xc) / len(X))
            counts = Xc.sum(axis=0) + self.alpha         # smoothed word counts
            self.log_lik[k] = np.log(counts / counts.sum())
        return self

    def predict(self, X):
        scores = X @ self.log_lik.T + self.log_prior     # log P(class | doc)
        return self.classes[scores.argmax(axis=1)]


class SoftmaxRegression:
    """Multinomial logistic regression trained with full-batch gradient descent."""

    def __init__(self, lr=0.5, epochs=300, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, d = X.shape
        C = len(self.classes)
        Y = (y[:, None] == self.classes[None, :]).astype(float)   # one-hot
        self.W = np.zeros((d, C))
        self.b = np.zeros(C)
        for _ in range(self.epochs):
            P = self._softmax(X @ self.W + self.b)
            G = P - Y                                    # dL/dz
            self.W -= self.lr * (X.T @ G / n + self.l2 * self.W)
            self.b -= self.lr * G.mean(axis=0)
        return self

    def predict(self, X):
        P = self._softmax(X @ self.W + self.b)
        return self.classes[P.argmax(axis=1)]


class CosineKNN:
    """k-Nearest Neighbors using cosine similarity on TF-IDF rows."""

    def __init__(self, k=7):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y                            # rows already L2-normalized
        self.classes = np.unique(y)
        return self

    def predict(self, X):
        sim = X @ self.X.T                               # cosine sim (unit rows)
        nn = np.argsort(-sim, axis=1)[:, :self.k]        # top-k neighbors
        votes = self.y[nn]
        out = np.zeros(len(X), dtype=self.y.dtype)
        for i, row in enumerate(votes):
            out[i] = np.bincount(row).argmax()
        return out


def make_reviews(n=900, seed=0):
    # Synthetic 3-class review corpus (0=negative, 1=neutral, 2=positive).
    # Each class draws mostly from its own sentiment vocabulary plus shared
    # filler, so the planted class signal is strong but overlapping/noisy.
    rng = np.random.RandomState(seed)
    topic = [
        ["awful", "terrible", "hate", "worst", "broken", "refund", "disappointed",
         "poor", "waste", "slow", "buggy", "annoying", "useless", "bad"],
        ["okay", "fine", "average", "decent", "normal", "fair", "mixed",
         "acceptable", "moderate", "meh", "standard", "ordinary"],
        ["great", "love", "excellent", "amazing", "best", "perfect", "fast",
         "reliable", "beautiful", "recommend", "fantastic", "smooth", "good"],
    ]
    shared = ["the", "a", "and", "is", "it", "this", "was", "for", "my", "very",
              "product", "app", "phone", "quite", "really", "with", "to", "of"]

    docs, labels = [], []
    for _ in range(n):
        c = rng.randint(3)                               # balanced-ish classes
        length = rng.randint(9, 18)
        n_core = int(length * 0.35)                       # 35% topical, 65% filler
        words = []
        for _ in range(n_core):
            # 25% of topical words leak in from a wrong class -> noisy, hard signal
            src = rng.randint(3) if rng.rand() < 0.25 else c
            words.append(rng.choice(topic[src]))
        words += list(rng.choice(shared, length - n_core))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(c)
    return docs, np.array(labels)


def macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return np.mean(f1s)


if __name__ == "__main__":
    np.random.seed(0)

    docs, y = make_reviews(n=900, seed=0)
    classes = np.unique(y)

    # Held-out split: train on 70%, test on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    tr_docs, te_docs = [docs[i] for i in tr], [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = BowVectorizer().fit(tr_docs)
    Xc_tr, Xc_te = vec.counts(tr_docs), vec.counts(te_docs)     # for Naive Bayes
    Xt_tr, Xt_te = vec.tfidf(tr_docs), vec.tfidf(te_docs)       # for LogReg / kNN

    # Comparison harness: (name, fitted model, held-out test matrix).
    models = [
        ("Multinomial NaiveBayes", MultinomialNB(alpha=1.0).fit(Xc_tr, ytr), Xc_te),
        ("Softmax LogRegression ", SoftmaxRegression().fit(Xt_tr, ytr), Xt_te),
        ("Cosine kNN (k=7)       ", CosineKNN(k=7).fit(Xt_tr, ytr), Xt_te),
    ]

    # Majority-class baseline: always predict the most common training label.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, classes)

    print("Reviews: %d   Train: %d   Test: %d   Classes: %d   Vocab: %d"
          % (len(docs), len(tr), len(te), len(classes), len(vec.vocab)))
    print("=" * 62)
    print("%-24s %10s %10s %8s" % ("Algorithm", "Accuracy", "MacroF1", "Beats?"))
    print("-" * 62)

    results = []
    for name, clf, Xte in models:
        pred = clf.predict(Xte)
        acc = np.mean(pred == yte)
        f1 = macro_f1(yte, pred, classes)
        beats = acc > base_acc and f1 > base_f1
        results.append((name, acc, f1))
        print("%-24s %10.4f %10.4f %8s" % (name, acc, f1, "yes" if beats else "NO"))

    print("-" * 62)
    print("%-24s %10.4f %10.4f %8s" % ("Majority baseline", base_acc, base_f1, "--"))
    print("=" * 62)

    winner = max(results, key=lambda r: (r[1], r[2]))
    print("Best algorithm: %s (accuracy %.4f)" % (winner[0].strip(), winner[1]))
    print("All algorithms beat baseline: %s"
          % all(a > base_acc and f > base_f1 for _, a, f in results))
