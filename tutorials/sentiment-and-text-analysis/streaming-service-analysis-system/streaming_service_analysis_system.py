import numpy as np


def tokenize(text):
    return text.lower().split()


class TfidfVectorizer:
    """TF-IDF from scratch. Learn a vocabulary and inverse-document-frequency
    weights, then turn each review into an L2-normalised TF-IDF vector so that
    words common to every review count for little and distinctive words weigh
    heavily."""

    def fit(self, docs):
        self.vocab = {}
        df = {}                                   # document frequency per word
        for d in docs:
            seen = set(tokenize(d))
            for w in seen:
                self.vocab.setdefault(w, len(self.vocab))
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf = np.ones(len(self.vocab))
        for w, j in self.vocab.items():
            self.idf[j] = np.log((1 + n) / (1 + df[w])) + 1.0   # smoothed idf
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)             # ignore unseen words
                if j is not None:
                    X[i, j] += 1.0                # raw term frequency
        X *= self.idf                             # weight by idf
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return X / norm                           # L2 normalise


class SoftmaxRegression:
    """Multinomial logistic regression trained by full-batch gradient descent.
    Scores are turned into class probabilities with a numerically stable
    softmax; weights are nudged down the cross-entropy gradient each step."""

    def __init__(self, lr=0.5, epochs=300, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def _softmax(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, d = X.shape
        k = len(self.classes)
        Y = np.zeros((n, k))                       # one-hot targets
        Y[np.arange(n), np.searchsorted(self.classes, y)] = 1.0
        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        for _ in range(self.epochs):
            P = self._softmax(X @ self.W + self.b)
            G = (P - Y) / n                         # cross-entropy gradient
            self.W -= self.lr * (X.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(axis=0)
        return self

    def predict(self, X):
        P = self._softmax(X @ self.W + self.b)
        return self.classes[np.argmax(P, axis=1)]


def make_reviews(n=1000, seed=0):
    # Synthetic streaming-service reviews. Each review mixes shared platform
    # vocabulary (no sentiment) with words from its own sentiment lexicon, so
    # the planted negative/neutral/positive signal is genuinely recoverable.
    rng = np.random.RandomState(seed)
    sentiment = {
        0: ["buffering", "crashes", "laggy", "overpriced", "cancel", "glitchy",
            "frustrating", "unwatchable", "refund", "terrible", "broken",
            "annoying", "worst", "downgrade"],                                    # negative
        1: ["okay", "average", "decent", "fine", "mixed", "watchable",
            "passable", "moderate", "forgettable", "middling", "acceptable"],     # neutral
        2: ["seamless", "binge", "addictive", "stunning", "affordable",
            "brilliant", "recommend", "flawless", "loved", "smooth",
            "immersive", "fantastic", "worth", "upgrade"],                        # positive
    }
    platform = ["stream", "subscription", "catalog", "episode", "series",
                "movie", "app", "playback", "resolution", "download", "profile",
                "watchlist", "originals", "documentary", "season", "trailer",
                "device", "login", "plan", "monthly", "library", "queue"]
    names = list(sentiment.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        length = rng.randint(10, 20)
        n_sent = max(3, int(length * 0.45))         # ~45% sentiment, rest platform
        words = list(rng.choice(sentiment[c], n_sent)) + \
            list(rng.choice(platform, length - n_sent))
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

    docs, y = make_reviews(n=1000, seed=0)
    label_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    # Held-out split: train on 70%, evaluate on the unseen 30%.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    tr_docs = [docs[i] for i in tr]
    te_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = TfidfVectorizer().fit(tr_docs)
    Xtr, Xte = vec.transform(tr_docs), vec.transform(te_docs)
    clf = SoftmaxRegression(lr=0.5, epochs=300).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, clf.classes)

    # Majority-class baseline: always guess the most common training label.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, clf.classes)

    print("Reviews: %d   Train: %d   Test: %d   Classes: %d"
          % (len(docs), len(tr), len(te), len(label_names)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 62)
    print("TF-IDF + Softmax accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class   accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 62)
    samples = [
        "buffering crashes overpriced cancel refund app playback plan monthly",
        "okay average decent watchable series catalog profile watchlist queue",
        "seamless binge addictive affordable loved originals stream download",
    ]
    for text in samples:
        c = clf.predict(vec.transform([text]))[0]
        print("  '%s...' -> %s" % (text[:42], label_names[c]))
    print("-" * 62)
    print("TF-IDF + Softmax beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
