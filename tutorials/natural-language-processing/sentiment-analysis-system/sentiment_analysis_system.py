import numpy as np


def tokenize(text):
    # Unigrams + bigrams so negations ("not good") become their own feature,
    # which is what lets the model separate polarity, not just word presence.
    words = text.lower().split()
    grams = list(words)
    grams += [words[i] + "_" + words[i + 1] for i in range(len(words) - 1)]
    return grams


class SentimentClassifier:
    """TF-IDF bag-of-n-grams + logistic regression trained by gradient descent."""

    def __init__(self, lr=1.0, epochs=600, l2=1e-5):
        self.lr = lr            # learning rate
        self.epochs = epochs    # full-batch gradient steps
        self.l2 = l2            # L2 regularization (keeps weights small)

    def _counts(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for g in tokenize(d):
                j = self.vocab.get(g)
                if j is not None:
                    X[i, j] += 1.0
        return X

    def _tfidf(self, X):
        # TF (row-normalized counts) weighted by fitted IDF -> down-weights
        # words common to both classes, up-weights discriminative ones.
        tf = X / np.maximum(X.sum(1, keepdims=True), 1.0)
        return tf * self.idf

    def fit(self, docs, y):
        y = np.asarray(y, dtype=float)
        self.vocab = {}
        for d in docs:
            for g in tokenize(d):
                self.vocab.setdefault(g, len(self.vocab))
        X = self._counts(docs)
        # IDF = log((1+N)/(1+df)) + 1, standard smoothed form.
        df = (X > 0).sum(0)
        self.idf = np.log((1 + X.shape[0]) / (1 + df)) + 1.0
        Xt = self._tfidf(X)

        n, d = Xt.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            z = Xt @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-z))            # sigmoid
            err = p - y
            gw = Xt.T @ err / n + self.l2 * self.w  # logistic loss gradient
            gb = err.mean()
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        return self

    def predict_proba(self, docs):
        Xt = self._tfidf(self._counts(docs))
        return 1.0 / (1.0 + np.exp(-(Xt @ self.w + self.b)))

    def predict(self, docs):
        return (self.predict_proba(docs) >= 0.5).astype(int)


def make_reviews(n=600, seed=0):
    # Synthetic reviews with planted polarity. A review's sentiment is expressed
    # EITHER by a direct polar word ("great") OR by negating the opposite word
    # ("not terrible"). Both classes reuse the same unigrams, so the negation
    # BIGRAMS ("not_great" vs "not_terrible") carry decisive signal.
    rng = np.random.RandomState(seed)
    pos = ["great", "love", "excellent", "amazing", "wonderful", "best",
           "fantastic", "brilliant", "enjoyed", "perfect", "recommend"]
    neg = ["terrible", "hate", "awful", "boring", "worst", "poor",
           "disappointing", "bad", "waste", "horrible", "avoid"]
    neutral = ["the", "a", "movie", "film", "story", "it", "was", "this",
               "and", "really", "quite", "very", "plot", "acting"]

    docs, labels = [], []
    for _ in range(n):
        positive = rng.rand() < 0.5                   # target sentiment
        same, opp = (pos, neg) if positive else (neg, pos)
        words = []
        length = rng.randint(9, 14)
        n_polar = max(3, int(length * 0.45))
        for _ in range(n_polar):
            if rng.rand() < 0.30:                      # negated opposite word
                words += ["not", rng.choice(opp)]      # e.g. "not terrible" = pos
            else:
                words.append(rng.choice(same))         # e.g. "great"        = pos
        words += list(rng.choice(neutral, length - n_polar))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(1 if positive else 0)           # 1 = positive, 0 = negative
    return docs, np.array(labels)


def metrics(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, f1


if __name__ == "__main__":
    np.random.seed(0)

    docs, y = make_reviews(n=600, seed=0)
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Xtr, Xte = [docs[i] for i in tr], [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    clf = SentimentClassifier().fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc, f1 = metrics(yte, pred)

    # Majority-class baseline: always predict the more common training label.
    majority = int(np.round(ytr.mean()))
    base_pred = np.full_like(yte, majority)
    base_acc, base_f1 = metrics(yte, base_pred)

    print("Reviews: %d   Train: %d   Test: %d   Positive rate: %.2f"
          % (len(docs), len(tr), len(te), y.mean()))
    print("Vocabulary size (uni+bigrams): %d" % len(clf.vocab))
    print("-" * 58)
    print("LogReg (TF-IDF)  accuracy: %.4f   F1(pos): %.4f" % (acc, f1))
    print("Majority class   accuracy: %.4f   F1(pos): %.4f" % (base_acc, base_f1))
    print("-" * 58)
    for text in ["a great wonderful enjoyed film",       # direct positive
                 "the plot was not great and boring",     # negated positive word
                 "not awful and really enjoyed the film"]:  # negated negative word
        p = clf.predict_proba([text])[0]
        print("  '%s' -> %s (p=%.2f)"
              % (text, "POSITIVE" if p >= 0.5 else "NEGATIVE", p))
    print("-" * 58)
    print("LogReg beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
