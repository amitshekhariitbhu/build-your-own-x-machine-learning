import numpy as np


def tokenize(text):
    return text.lower().split()


class HashingVectorizer:
    """Feature hashing (the 'hashing trick'). A real-time system cannot pause
    to build a vocabulary, so every token is mapped to a fixed bucket by a
    hash. This lets us featurize a message the instant it arrives -- no fit()
    pass over the corpus, unbounded vocabulary, constant memory."""

    def __init__(self, dim=4096):
        self.dim = dim

    @staticmethod
    def _hash(w):
        h = 2166136261                                # FNV-1a: deterministic + fast
        for ch in w:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        return h

    def transform_one(self, text):
        x = np.zeros(self.dim)
        for w in tokenize(text):
            h = self._hash(w)
            j = h % self.dim
            x[j] += 1.0 if (h >> 20) & 1 else -1.0   # signed hashing curbs collisions
        return x


class OnlineLogisticRegression:
    """Binary logistic regression trained by streaming SGD. partial_fit() does
    a single gradient step on one example, so the model improves message by
    message as the stream flows -- the core of a real-time learner."""

    def __init__(self, dim=4096, lr=0.5, l2=1e-4):
        self.w = np.zeros(dim)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict_proba(self, x):
        return self._sigmoid(x @ self.w + self.b)

    def predict(self, x):
        return int(self.predict_proba(x) >= 0.5)

    def partial_fit(self, x, y):
        p = self.predict_proba(x)
        g = p - y                                     # dL/dz for log-loss
        self.w -= self.lr * (g * x + self.l2 * self.w)
        self.b -= self.lr * g


def make_stream(n=2000, seed=0):
    # Synthetic chat/review stream. Each message is POSITIVE (1) or NEGATIVE (0)
    # drawn mostly from its own opinion words plus shared neutral filler, so the
    # planted class signal is real but words overlap across classes.
    rng = np.random.RandomState(seed)
    pos = ["love", "great", "excellent", "amazing", "smooth", "best", "perfect",
           "fast", "beautiful", "helpful", "awesome", "fantastic", "happy", "wow"]
    neg = ["crash", "bug", "terrible", "worst", "broken", "freezes", "useless",
           "annoying", "laggy", "refund", "awful", "hate", "slow", "disappointed"]
    filler = ["the", "app", "update", "phone", "today", "again", "really", "just",
              "it", "this", "and", "so", "was", "now", "screen", "version"]
    docs, labels = [], []
    for _ in range(n):
        c = rng.randint(2)
        length = rng.randint(6, 14)
        n_op = max(2, int(length * 0.5))              # ~50% opinion, rest filler
        pool = pos if c == 1 else neg
        words = list(rng.choice(pool, n_op)) + \
            list(rng.choice(filler, length - n_op))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(c)
    return docs, np.array(labels)


if __name__ == "__main__":
    np.random.seed(0)

    docs, y = make_stream(n=2000, seed=0)

    # Hold out the last 25% of the stream as unseen future messages.
    split = int(0.75 * len(docs))
    stream_docs, stream_y = docs[:split], y[:split]
    test_docs, test_y = docs[split:], y[split:]

    vec = HashingVectorizer(dim=4096)
    clf = OnlineLogisticRegression(dim=4096, lr=0.5)

    # REAL-TIME LOOP: prequential (test-then-train) evaluation. For every
    # arriving message we first predict (as we would live), score it, then
    # learn from it. The running accuracy shows the model improving on the fly.
    correct, seen, checkpoints = 0, 0, []
    for text, label in zip(stream_docs, stream_y):
        x = vec.transform_one(text)
        correct += int(clf.predict(x) == label)
        seen += 1
        clf.partial_fit(x, label)
        if seen % 300 == 0:
            checkpoints.append((seen, correct / seen))

    # Final evaluation on the held-out future messages (no more learning).
    preds = np.array([clf.predict(vec.transform_one(t)) for t in test_docs])
    acc = np.mean(preds == test_y)
    tp = np.sum((preds == 1) & (test_y == 1))
    fp = np.sum((preds == 1) & (test_y == 0))
    fn = np.sum((preds == 0) & (test_y == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    # Majority-class baseline on the same held-out set.
    majority = np.bincount(stream_y).argmax()
    base_acc = np.mean(test_y == majority)

    print("Stream messages: %d   Trained on: %d   Held-out: %d"
          % (len(docs), len(stream_docs), len(test_docs)))
    print("Feature dim (hashed): %d" % vec.dim)
    print("-" * 60)
    print("Real-time running accuracy (prequential):")
    for s, a in checkpoints:
        print("  after %4d msgs: %.4f" % (s, a))
    print("-" * 60)
    print("Online LogReg  held-out accuracy: %.4f   F1: %.4f" % (acc, f1))
    print("Majority class held-out accuracy: %.4f" % base_acc)
    print("-" * 60)
    samples = ["love this update smooth and fast best app",
               "worst app it keeps crashing laggy refund please",
               "amazing helpful and beautiful screen wow",
               "terrible broken freezes again hate this version"]
    for text in samples:
        p = clf.predict_proba(vec.transform_one(text))
        tag = "POSITIVE" if p >= 0.5 else "NEGATIVE"
        print("  [%s p=%.2f] '%s'" % (tag, p, text[:38]))
    print("-" * 60)
    print("Online model beats majority baseline: %s" % (acc > base_acc))
