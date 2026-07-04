import numpy as np


def tokenize(text):
    return text.lower().split()


class CountVectorizer:
    """Bag-of-words counts from scratch: build a vocabulary, then turn each
    document into a vector of raw term frequencies over that vocabulary."""

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
                j = self.vocab.get(w)          # ignore unseen words
                if j is not None:
                    X[i, j] += 1.0
        return X


class MultinomialNB:
    """Multinomial Naive Bayes from scratch. Learns per-class word likelihoods
    with Laplace smoothing, then scores documents in log space (adding logs
    instead of multiplying probabilities to avoid underflow)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha                     # Laplace smoothing strength

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, v = X.shape
        self.log_prior = np.zeros(len(self.classes))
        self.log_likelihood = np.zeros((len(self.classes), v))
        for k, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[k] = np.log(len(Xc) / n)            # P(class)
            counts = Xc.sum(0) + self.alpha                    # smoothed word counts
            self.log_likelihood[k] = np.log(counts / counts.sum())  # P(word|class)
        return self

    def predict_log_proba(self, X):
        # log P(class) + sum over words of count * log P(word|class).
        scores = X @ self.log_likelihood.T + self.log_prior
        scores -= scores.max(axis=1, keepdims=True)            # stabilise
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)
        return np.log(probs)

    def predict(self, X):
        return self.classes[np.argmax(self.predict_log_proba(X), axis=1)]


def make_reviews(n=900, seed=0):
    # Synthetic Squid Game reviews. Every review mixes shared show vocabulary
    # (no sentiment) with words drawn from its own sentiment lexicon, so the
    # planted positive/neutral/negative signal is genuinely recoverable.
    rng = np.random.RandomState(seed)
    sentiment = {
        0: ["boring", "predictable", "disappointing", "overrated", "dull",
            "hated", "slow", "waste", "terrible", "weak", "cringe", "meh"],       # negative
        1: ["okay", "average", "watchable", "fine", "decent", "mixed",
            "passable", "moderate", "forgettable", "middling"],                    # neutral
        2: ["masterpiece", "gripping", "brilliant", "addictive", "stunning",
            "loved", "amazing", "thrilling", "flawless", "perfect", "intense"],    # positive
    }
    show = ["squid", "game", "players", "guard", "mask", "green", "tracksuit",
            "456", "marbles", "tug", "war", "gihun", "frontman", "money",
            "prize", "elimination", "netflix", "korean", "drama", "season",
            "doll", "redlight", "greenlight", "dalgona", "vote", "island"]
    names = list(sentiment.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        length = rng.randint(10, 20)
        n_sent = max(3, int(length * 0.45))          # ~45% sentiment, rest show terms
        words = list(rng.choice(sentiment[c], n_sent)) + \
            list(rng.choice(show, length - n_sent))
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

    docs, y = make_reviews(n=900, seed=0)
    label_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    # Held-out split: train on 70%, evaluate on the unseen 30%.
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
    print("Naive Bayes    accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 62)
    samples = [
        "squid game masterpiece gripping addictive thrilling netflix season",
        "boring predictable overrated waste guard mask elimination dull",
        "okay average watchable decent players marbles money vote fine",
    ]
    for text in samples:
        c = clf.predict(vec.transform([text]))[0]
        print("  '%s...' -> %s" % (text[:40], label_names[c]))
    print("-" * 62)
    print("Naive Bayes beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
