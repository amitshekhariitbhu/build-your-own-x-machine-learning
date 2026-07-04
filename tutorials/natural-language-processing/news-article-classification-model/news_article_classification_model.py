import numpy as np


def tokenize(text):
    return text.lower().split()


class TfidfVectorizer:
    """Bag-of-words -> TF-IDF matrix, all fit from the training corpus."""

    def fit(self, docs):
        # Vocabulary = every word seen in training.
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        # Document frequency -> smoothed inverse-document-frequency weights.
        df = np.zeros(len(self.vocab))
        for d in docs:
            for w in set(tokenize(d)):
                df[self.vocab[w]] += 1.0
        n = len(docs)
        self.idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] += 1.0                 # raw term counts (TF)
        X *= self.idf[None, :]                     # weight rare words higher
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm == 0, 1.0, norm)  # L2-normalise each doc


class SoftmaxClassifier:
    """Multinomial logistic regression trained by full-batch gradient descent."""

    def __init__(self, lr=0.5, epochs=300, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        n, d = X.shape
        self.classes = np.unique(y)
        k = len(self.classes)
        Y = np.zeros((n, k))                        # one-hot targets
        Y[np.arange(n), np.searchsorted(self.classes, y)] = 1.0
        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        for _ in range(self.epochs):
            P = self._softmax(X @ self.W + self.b)
            gW = X.T @ (P - Y) / n + self.l2 * self.W
            gb = (P - Y).mean(0)
            self.W -= self.lr * gW
            self.b -= self.lr * gb
        return self

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes[np.argmax(X @ self.W + self.b, axis=1)]


def make_news(n=800, seed=0):
    # Synthetic newswire: each category draws mostly from its own topic words
    # plus shared filler, so the planted topic signal is recoverable but noisy.
    rng = np.random.RandomState(seed)
    topics = {
        0: ["election", "senate", "policy", "government", "vote", "minister", "law", "campaign"],
        1: ["match", "goal", "team", "coach", "season", "player", "score", "championship"],
        2: ["software", "chip", "startup", "device", "algorithm", "data", "app", "launch"],
        3: ["market", "profit", "shares", "revenue", "economy", "trade", "investors", "quarter"],
    }
    shared = ["the", "said", "today", "reported", "after", "will", "new", "year", "on", "a"]
    names = list(topics.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        length = rng.randint(12, 22)
        n_topic = int(length * 0.6)                 # 60% topical, 40% filler
        words = list(rng.choice(topics[c], n_topic)) + \
            list(rng.choice(shared, length - n_topic))
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

    docs, y = make_news(n=800, seed=0)
    categories = ["Politics", "Sports", "Technology", "Business"]

    # Held-out split: train on 70%, test on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Xtr_docs = [docs[i] for i in tr]
    Xte_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = TfidfVectorizer().fit(Xtr_docs)
    Xtr, Xte = vec.transform(Xtr_docs), vec.transform(Xte_docs)
    clf = SoftmaxClassifier().fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, clf.classes)

    # Majority-class baseline: always predict the most common training label.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, clf.classes)

    print("Articles: %d   Train: %d   Test: %d   Classes: %d"
          % (len(docs), len(tr), len(te), len(categories)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 58)
    print("Softmax TF-IDF  accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class  accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 58)
    samples = ["senate vote election policy government today",
               "team goal coach match season player",
               "startup chip software algorithm launch data",
               "market profit shares revenue investors quarter"]
    for text in samples:
        c = clf.predict(vec.transform([text]))[0]
        print("  '%s...' -> %s" % (text[:34], categories[c]))
    print("-" * 58)
    print("Softmax beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
