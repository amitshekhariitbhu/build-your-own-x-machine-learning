import numpy as np


def tokenize(text):
    return text.lower().split()


class TfidfVectorizer:
    """Bag-of-words -> TF-IDF matrix, fit on training vocabulary only."""

    def fit(self, docs):
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        # Document frequency per word -> smoothed inverse document frequency.
        df = np.zeros(len(self.vocab))
        for d in docs:
            for w in set(tokenize(d)):
                df[self.vocab[w]] += 1.0
        self.idf = np.log((1.0 + len(docs)) / (1.0 + df)) + 1.0
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] += 1.0                      # term frequency (counts)
        X *= self.idf[None, :]                          # weight by IDF
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm == 0, 1.0, norm)       # L2-normalize rows

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)


class LogisticRegression:
    """Binary logistic regression trained by full-batch gradient descent."""

    def __init__(self, lr=0.5, epochs=300, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y                                 # gradient of log-loss
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def make_dataset(n=700, seed=0):
    # Synthetic moderation corpus: harmful vs safe messages drawn from
    # overlapping vocabularies but distinct word frequencies -> separable signal.
    rng = np.random.RandomState(seed)
    harmful = ["hate", "kill", "attack", "threat", "stupid", "idiot",
               "destroy", "violence", "abuse", "worthless", "die", "hurt"]
    safe = ["hello", "please", "thanks", "great", "help", "welcome",
            "share", "learn", "kind", "support", "friend", "happy"]
    shared = ["you", "the", "this", "and", "people", "today", "here", "we"]

    docs, labels = [], []
    for _ in range(n):
        bad = rng.rand() < 0.4                          # 40% harmful, 60% safe
        topic = harmful if bad else safe
        length = rng.randint(8, 16)
        n_topic = int(length * 0.6)                     # topic words + filler
        words = list(rng.choice(topic, n_topic)) + \
            list(rng.choice(shared, length - n_topic))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(1 if bad else 0)                  # 1 = harmful, 0 = safe
    return docs, np.array(labels)


def metrics(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, prec, rec, f1


def roc_auc(y_true, scores):
    # AUC = probability a random positive scores above a random negative.
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = y_true.sum(), (1 - y_true).sum()
    if pos == 0 or neg == 0:
        return 0.5
    return (ranks[y_true == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


if __name__ == "__main__":
    np.random.seed(0)

    docs, y = make_dataset(n=700, seed=0)
    # Held-out split: train on 70%, evaluate on the untouched 30%.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Dtr, Dte = [docs[i] for i in tr], [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = TfidfVectorizer().fit(Dtr)
    Xtr, Xte = vec.transform(Dtr), vec.transform(Dte)

    clf = LogisticRegression(lr=0.5, epochs=300).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)
    acc, prec, rec, f1 = metrics(yte, pred)
    auc = roc_auc(yte, prob)

    # Majority-class baseline: always predict the more common training label.
    majority = int(np.round(ytr.mean()))
    base_pred = np.full_like(yte, majority)
    base_acc, base_prec, base_rec, base_f1 = metrics(yte, base_pred)

    print("Messages: %d   Train: %d   Test: %d   Harmful rate: %.2f"
          % (len(docs), len(tr), len(te), y.mean()))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 60)
    print("TF-IDF + LogReg  accuracy: %.4f  F1: %.4f  AUC: %.4f"
          % (acc, f1, auc))
    print("Majority class   accuracy: %.4f  F1: %.4f  AUC: %.4f"
          % (base_acc, base_f1, 0.5))
    print("-" * 60)
    for text in ["i hate you idiot go die", "hello friend thanks for the help"]:
        p = clf.predict(vec.transform([text]))[0]
        s = clf.predict_proba(vec.transform([text]))[0]
        print("  '%s' -> %s (%.2f)" % (text, "HARMFUL" if p else "SAFE", s))
    print("-" * 60)
    print("Beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1 and auc > 0.5))
