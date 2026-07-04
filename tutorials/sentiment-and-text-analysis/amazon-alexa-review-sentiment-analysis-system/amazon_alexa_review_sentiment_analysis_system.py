import numpy as np


def tokenize(text):
    return text.lower().split()


class TfidfVectorizer:
    """TF-IDF features from scratch. Builds a vocabulary from the training
    reviews, then weights each term by its frequency in a review (TF) scaled by
    how rare the term is across the corpus (IDF), so common filler words get
    down-weighted and discriminative words stand out."""

    def fit(self, docs):
        self.vocab = {}
        df = {}                                # document frequency per word
        for d in docs:
            for w in set(tokenize(d)):
                self.vocab.setdefault(w, len(self.vocab))
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf = np.ones(len(self.vocab))
        for w, j in self.vocab.items():
            self.idf[j] = np.log((1 + n) / (1 + df[w])) + 1.0   # smoothed IDF
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            toks = tokenize(d)
            for w in toks:
                j = self.vocab.get(w)          # ignore unseen words
                if j is not None:
                    X[i, j] += 1.0
            if toks:
                X[i] /= len(toks)              # term frequency (normalised)
        X *= self.idf                          # apply inverse doc frequency
        # L2-normalise rows so review length does not dominate the score.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms


class LogisticRegression:
    """Binary logistic regression trained with full-batch gradient descent and
    L2 regularisation. Learns a weight per vocabulary word plus a bias, then
    predicts P(positive) = sigmoid(w . x + b)."""

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
            err = p - y                        # gradient of log-loss
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def make_alexa_reviews(n=1400, seed=0):
    # Synthetic Amazon Alexa reviews. Positive (feedback=1) reviews praise the
    # device; negative (feedback=0) reviews complain. Each class draws mostly
    # from its own opinion words plus shared Alexa/device chatter, planting a
    # sentiment signal recoverable from word weights alone.
    rng = np.random.RandomState(seed)
    positive = ["love", "great", "easy", "perfect", "amazing", "works",
                "awesome", "happy", "excellent", "convenient", "helpful",
                "reliable", "recommend", "fantastic", "impressed", "enjoy"]
    negative = ["disappointed", "useless", "stopped", "waste", "broken",
                "terrible", "poor", "refund", "frustrating", "returned",
                "faulty", "junk", "unresponsive", "glitchy", "regret", "awful"]
    shared = ["alexa", "echo", "dot", "device", "music", "the", "and", "it",
              "for", "my", "this", "with", "speaker", "sound", "home"]

    docs, labels = [], []
    for _ in range(n):
        c = rng.randint(2)                     # 0 = negative, 1 = positive
        length = rng.randint(10, 18)
        n_op = max(3, int(length * 0.5))       # 50% opinion, 50% chatter
        pool = positive if c == 1 else negative
        words = list(rng.choice(pool, n_op)) + \
            list(rng.choice(shared, length - n_op))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(c)
    return docs, np.array(labels)


def prf1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)

    docs, y = make_alexa_reviews(n=1400, seed=0)

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
    prec, rec, f1 = prf1(yte, pred)

    # Majority-class baseline: always predict the most common training feedback.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    _, _, base_f1 = prf1(yte, base_pred)

    print("Reviews: %d   Train: %d   Test: %d   Classes: 2 (neg/pos)"
          % (len(docs), len(tr), len(te)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 60)
    print("LogReg TF-IDF  accuracy: %.4f  precision: %.4f  recall: %.4f  F1: %.4f"
          % (acc, prec, rec, f1))
    print("Majority class accuracy: %.4f  F1: %.4f" % (base_acc, base_f1))
    print("-" * 60)
    samples = ["alexa echo dot love great easy perfect music works recommend",
               "alexa device stopped useless waste broken refund returned junk"]
    labels_txt = ["negative", "positive"]
    for text in samples:
        p = clf.predict_proba(vec.transform([text]))[0]
        c = int(p >= 0.5)
        print("  P(pos)=%.2f -> %s : '%s...'" % (p, labels_txt[c], text[:38]))
    print("-" * 60)
    print("LogReg beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
