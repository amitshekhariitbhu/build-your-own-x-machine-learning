import numpy as np


def tokenize(text):
    return text.lower().split()


class CountVectorizer:
    """Bag-of-words counts from scratch. Learns a vocabulary from the training
    reviews, then turns each review into a vector of raw word counts, one slot
    per vocabulary word (unseen words are ignored)."""

    def fit(self, docs):
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] += 1.0
        return X


class MultinomialNB:
    """Multinomial Naive Bayes from scratch. For each class it estimates a word
    distribution with Laplace smoothing and a class prior, then scores a review
    in log-space: log P(class) + sum(count_w * log P(w | class)). The argmax
    over classes is the prediction. Log-space keeps the products stable."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, d = X.shape
        self.log_prior = np.zeros(len(self.classes))
        self.log_likelihood = np.zeros((len(self.classes), d))
        for k, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[k] = np.log(len(Xc) / n)
            counts = Xc.sum(axis=0) + self.alpha        # smoothed word counts
            self.log_likelihood[k] = np.log(counts / counts.sum())
        return self

    def predict(self, X):
        # scores[i, k] = log P(class_k) + sum_w X[i,w] * log P(w | class_k)
        scores = X @ self.log_likelihood.T + self.log_prior
        return self.classes[scores.argmax(axis=1)]


def make_amazon_reviews(n=1800, seed=0):
    # Synthetic Amazon product reviews across 3 sentiments keyed to star ratings:
    # 0 = negative (1-2 stars), 1 = neutral (3 stars), 2 = positive (4-5 stars).
    # Each class draws mostly from its own opinion words plus shared product
    # chatter, planting a sentiment signal recoverable from word counts alone.
    rng = np.random.RandomState(seed)
    negative = ["terrible", "broke", "disappointed", "cheap", "waste", "refund",
                "defective", "returned", "poor", "useless", "awful", "flimsy",
                "stopped", "worst", "regret", "frustrating"]
    neutral = ["okay", "average", "decent", "fine", "expected", "alright",
               "mediocre", "acceptable", "ordinary", "unremarkable", "fair",
               "middling", "adequate", "so-so", "passable", "moderate"]
    positive = ["love", "excellent", "great", "perfect", "amazing", "durable",
                "recommend", "fantastic", "quality", "worth", "sturdy", "happy",
                "impressed", "reliable", "wonderful", "best"]
    shared = ["product", "item", "the", "and", "it", "for", "my", "this",
              "with", "shipping", "price", "packaging", "arrived", "box", "order"]

    pools = [negative, neutral, positive]
    docs, labels = [], []
    for _ in range(n):
        c = rng.randint(3)
        length = rng.randint(12, 20)
        n_op = max(4, int(length * 0.55))               # 55% opinion words
        words = list(rng.choice(pools[c], n_op)) + \
            list(rng.choice(shared, length - n_op))
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

    docs, y = make_amazon_reviews(n=1800, seed=0)
    classes = np.array([0, 1, 2])
    names = ["negative", "neutral", "positive"]

    # Held-out split: train on 70%, test on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    tr_docs = [docs[i] for i in tr]
    te_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = CountVectorizer().fit(tr_docs)
    Xtr, Xte = vec.transform(tr_docs), vec.transform(te_docs)
    clf = MultinomialNB(alpha=1.0).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, classes)

    # Majority-class baseline: always predict the most common training class.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, classes)

    print("Reviews: %d   Train: %d   Test: %d   Classes: 3 (neg/neu/pos)"
          % (len(docs), len(tr), len(te)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 62)
    print("MultinomialNB  accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 62)
    samples = ["product arrived broke terrible waste refund returned defective",
               "the item is okay average decent fine for the price expected",
               "love this product excellent quality durable worth recommend best"]
    for text in samples:
        c = int(clf.predict(vec.transform([text]))[0])
        print("  %-9s : '%s...'" % (names[c], text[:40]))
    print("-" * 62)
    print("MultinomialNB beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
