import numpy as np


def tokenize(text):
    return text.lower().split()


class TfidfVectorizer:
    """Bag-of-words with TF-IDF weighting, from scratch. We learn a vocabulary
    and document frequencies on train, then map each review to a vector of
    tf * idf values (idf = log((1+N)/(1+df)) + 1), L2-normalized per row."""

    def fit(self, docs):
        self.vocab = {}
        df = {}
        for d in docs:
            for w in set(tokenize(d)):
                self.vocab.setdefault(w, len(self.vocab))
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf = np.ones(len(self.vocab))
        for w, j in self.vocab.items():
            self.idf[j] = np.log((1 + n) / (1 + df[w])) + 1.0
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)          # ignore words unseen in training
                if j is not None:
                    X[i, j] += 1.0             # term frequency
        X *= self.idf                          # apply inverse doc frequency
        norm = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
        return X / np.where(norm == 0, 1.0, norm)   # L2 normalize rows


class LogisticRegression:
    """Binary logistic regression trained with full-batch gradient descent and
    L2 regularization. Predicts P(positive) via the sigmoid of a linear score."""

    def __init__(self, lr=0.5, epochs=300, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def make_reviews(n=1000, seed=0):
    # Synthetic hotel reviews. Each review draws mostly from its class opinion
    # words plus shared hotel filler, so the planted signal is real but words
    # overlap across classes -- a recoverable structure, not pure noise.
    rng = np.random.RandomState(seed)
    positive = ["clean", "spacious", "comfortable", "friendly", "helpful",
                "spotless", "quiet", "excellent", "lovely", "delicious",
                "cozy", "stunning", "welcoming", "value", "recommend"]
    negative = ["dirty", "noisy", "rude", "smelly", "cramped", "cold",
                "overpriced", "broken", "filthy", "disappointing", "stained",
                "unhelpful", "outdated", "roaches", "awful"]
    shared = ["hotel", "room", "stay", "night", "staff", "breakfast", "bed",
              "location", "check-in", "pool", "the", "was", "and", "we", "our"]

    docs, labels = [], []
    for _ in range(n):
        c = rng.randint(2)                      # 0 negative, 1 positive
        length = rng.randint(10, 20)
        n_op = max(4, int(length * 0.5))        # ~50% opinion, rest filler
        pool = positive if c == 1 else negative
        words = list(rng.choice(pool, n_op)) + \
            list(rng.choice(shared, length - n_op))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(c)
    return docs, np.array(labels)


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

    docs, y = make_reviews(n=1000, seed=0)

    # Held-out split: train on 70%, test on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    tr_docs = [docs[i] for i in tr]
    te_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = TfidfVectorizer().fit(tr_docs)
    Xtr, Xte = vec.transform(tr_docs), vec.transform(te_docs)
    clf = LogisticRegression(lr=0.5, epochs=300, l2=1e-3).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    prec, rec, f1 = prf(yte, pred)

    # Majority-class baseline: always predict the most common training class.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    _, _, base_f1 = prf(yte, base_pred)

    print("Reviews: %d   Train: %d   Test: %d   Classes: 2 (NEG/POS)"
          % (len(docs), len(tr), len(te)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 62)
    print("Logistic Reg   accuracy: %.4f   P: %.3f  R: %.3f  F1: %.4f"
          % (acc, prec, rec, f1))
    print("Majority class accuracy: %.4f   F1: %.4f" % (base_acc, base_f1))
    print("-" * 62)
    samples = ["the room was dirty noisy and the staff were rude",
               "clean spacious room friendly helpful staff great value",
               "smelly cramped overpriced broken bed awful stay",
               "lovely quiet hotel spotless comfortable bed delicious breakfast"]
    for text in samples:
        p = clf.predict_proba(vec.transform([text]))[0]
        tag = "POSITIVE" if p >= 0.5 else "NEGATIVE"
        print("  P(pos)=%.2f -> %-8s | '%s...'" % (p, tag, text[:40]))
    print("-" * 62)
    print("Logistic Reg beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
