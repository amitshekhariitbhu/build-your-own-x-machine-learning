import numpy as np


def tokenize(text):
    return text.lower().split()


class TfidfVectorizer:
    """Bag-of-words -> TF-IDF from scratch: term counts scaled by inverse
    document frequency so common filler words carry less weight."""

    def fit(self, docs):
        self.vocab = {}
        for d in docs:
            for w in set(tokenize(d)):
                self.vocab.setdefault(w, len(self.vocab))
        # Smoothed IDF: log((1 + N) / (1 + df)) + 1, like the textbook form.
        df = np.zeros(len(self.vocab))
        for d in docs:
            for w in set(tokenize(d)):
                df[self.vocab[w]] += 1.0
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1.0
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)          # ignore unseen words
                if j is not None:
                    X[i, j] += 1.0
        X *= self.idf                          # term freq * inverse doc freq
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms == 0, 1.0, norms)   # L2-normalise rows


class SoftmaxRegression:
    """Multinomial logistic regression trained with full-batch gradient
    descent. Softmax over class scores; cross-entropy loss; manual gradients."""

    def __init__(self, lr=0.5, epochs=300, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def _softmax(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True)   # stabilise before exp
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        n, d = X.shape
        self.classes = np.unique(y)
        k = len(self.classes)
        Y = np.zeros((n, k))                    # one-hot targets
        Y[np.arange(n), np.searchsorted(self.classes, y)] = 1.0
        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        for _ in range(self.epochs):
            P = self._softmax(X @ self.W + self.b)
            G = (P - Y) / n                      # gradient of cross-entropy
            self.W -= self.lr * (X.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(0)
        return self

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]

    def predict_topk(self, X, k=2):
        # Indices of the k highest-probability hashtags per tweet.
        order = np.argsort(-self.predict_proba(X), axis=1)[:, :k]
        return self.classes[order]


def make_tweets(n=1000, seed=0):
    # Synthetic tweets: each hashtag samples mostly from its own topic words
    # plus shared chatter, so the planted hashtag signal is recoverable.
    rng = np.random.RandomState(seed)
    topics = {
        0: ["goal", "match", "team", "score", "coach", "playoffs", "league", "win"],
        1: ["gpu", "startup", "app", "code", "ai", "chip", "software", "launch"],
        2: ["recipe", "pizza", "coffee", "dinner", "tasty", "chef", "brunch", "vegan"],
        3: ["guitar", "album", "concert", "song", "band", "remix", "vinyl", "lyrics"],
        4: ["flight", "beach", "hotel", "trip", "passport", "island", "hiking", "sunset"],
    }
    shared = ["the", "a", "just", "so", "this", "my", "we", "lol", "today", "vibes"]
    names = list(topics.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        length = rng.randint(8, 16)
        n_topic = max(2, int(length * 0.5))          # 50% topical, 50% chatter
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

    docs, y = make_tweets(n=1000, seed=0)
    hashtags = ["#sports", "#tech", "#food", "#music", "#travel"]

    # Held-out split: train on 70%, test on the rest.
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
    top2 = clf.predict_topk(Xte, k=2)
    top2_acc = np.mean([yt in row for yt, row in zip(yte, top2)])

    # Majority-class baseline: always predict the most common training hashtag.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, clf.classes)

    print("Tweets: %d   Train: %d   Test: %d   Hashtags: %d"
          % (len(docs), len(tr), len(te), len(hashtags)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 60)
    print("Softmax model  accuracy: %.4f   macro-F1: %.4f   top-2 acc: %.4f"
          % (acc, f1, top2_acc))
    print("Majority class accuracy: %.4f   macro-F1: %.4f"
          % (base_acc, base_f1))
    print("-" * 60)
    samples = ["team score goal playoffs coach lol",
               "startup launch ai app chip code today",
               "recipe pizza coffee tasty chef brunch",
               "concert album guitar band vinyl song",
               "flight hotel beach trip passport sunset"]
    for text in samples:
        c = clf.predict(vec.transform([text]))[0]
        print("  '%s...' -> %s" % (text[:34], hashtags[c]))
    print("-" * 60)
    print("Softmax model beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
