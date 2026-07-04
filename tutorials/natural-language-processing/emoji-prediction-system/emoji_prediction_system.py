import numpy as np


def tokenize(text):
    return text.lower().split()


class EmojiPredictor:
    """Predict an emoji for a message: TF-IDF bag-of-words + softmax regression
    (multinomial logistic regression) trained from scratch by gradient descent."""

    def __init__(self, lr=0.5, epochs=300, l2=1e-4):
        self.lr = lr            # learning rate
        self.epochs = epochs    # full-batch gradient steps
        self.l2 = l2            # L2 regularization keeps weights small

    def _counts(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] += 1.0
        return X

    def _tfidf(self, X):
        # TF (row-normalized counts) weighted by fitted IDF -> down-weights
        # filler shared across emojis, up-weights topic-specific words.
        tf = X / np.maximum(X.sum(1, keepdims=True), 1.0)
        return tf * self.idf

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)          # stabilize before exp
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def fit(self, docs, y):
        y = np.asarray(y)
        self.classes = np.unique(y)
        K = len(self.classes)
        cls_idx = {c: k for k, c in enumerate(self.classes)}

        # Vocabulary = every word seen in training.
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        X = self._counts(docs)
        # Smoothed IDF = log((1+N)/(1+df)) + 1.
        df = (X > 0).sum(0)
        self.idf = np.log((1 + X.shape[0]) / (1 + df)) + 1.0
        Xt = self._tfidf(X)

        n, V = Xt.shape
        Y = np.zeros((n, K))                      # one-hot targets
        Y[np.arange(n), [cls_idx[c] for c in y]] = 1.0

        self.W = np.zeros((V, K))
        self.b = np.zeros(K)
        for _ in range(self.epochs):
            P = self._softmax(Xt @ self.W + self.b)
            err = P - Y                           # softmax cross-entropy gradient
            self.W -= self.lr * (Xt.T @ err / n + self.l2 * self.W)
            self.b -= self.lr * err.mean(0)
        return self

    def predict_proba(self, docs):
        Xt = self._tfidf(self._counts(docs))
        return self._softmax(Xt @ self.W + self.b)

    def predict(self, docs):
        return self.classes[np.argmax(self.predict_proba(docs), axis=1)]


# Each emoji is a class with its own topic vocabulary; a shared filler set
# overlaps every class so the model must lean on the discriminative words.
EMOJI_TOPICS = {
    "\U0001F602": ["funny", "lol", "hilarious", "joke", "laughing", "haha"],   # laugh
    "❤️": ["love", "heart", "adore", "sweetheart", "darling", "hugs"],  # heart
    "\U0001F622": ["sad", "crying", "tears", "miss", "lonely", "heartbroken"],  # cry
    "\U0001F525": ["fire", "hype", "awesome", "amazing", "lit", "epic"],        # fire
    "\U0001F389": ["party", "celebrate", "congrats", "birthday", "cheers", "win"],  # party
    "\U0001F355": ["pizza", "hungry", "dinner", "tasty", "food", "eat"],        # food
}
SHARED = ["the", "a", "this", "is", "so", "really", "today", "you", "we", "just"]


def make_messages(n=900, seed=0):
    # Synthetic chat messages: pick an emoji, draw mostly its topic words plus
    # some shared filler -> overlapping but recoverable per-emoji signal.
    rng = np.random.RandomState(seed)
    emojis = list(EMOJI_TOPICS)
    docs, labels = [], []
    for _ in range(n):
        e = emojis[rng.randint(len(emojis))]
        topic = EMOJI_TOPICS[e]
        length = rng.randint(6, 12)
        n_topic = max(2, int(length * 0.6))
        words = list(rng.choice(topic, n_topic)) + \
            list(rng.choice(SHARED, length - n_topic))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(e)
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

    docs, y = make_messages(n=900, seed=0)
    # Held-out split: train on 70%, test on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Xtr = [docs[i] for i in tr]
    Xte = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    clf = EmojiPredictor().fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, clf.classes)

    # Majority-class baseline: always predict the most common training emoji.
    vals, cnts = np.unique(ytr, return_counts=True)
    majority = vals[np.argmax(cnts)]
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, clf.classes)

    print("Messages: %d   Train: %d   Test: %d   Emojis: %d"
          % (len(docs), len(tr), len(te), len(clf.classes)))
    print("Vocabulary size: %d   Random-guess accuracy: %.4f"
          % (len(clf.vocab), 1.0 / len(clf.classes)))
    print("-" * 58)
    print("Softmax (TF-IDF)  accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class    accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 58)
    for text in ["that joke was so hilarious lol", "i love you sweetheart",
                 "hungry lets get pizza for dinner", "congrats on the win party"]:
        print("  '%s' -> %s" % (text, clf.predict([text])[0]))
    print("-" * 58)
    print("Softmax beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
