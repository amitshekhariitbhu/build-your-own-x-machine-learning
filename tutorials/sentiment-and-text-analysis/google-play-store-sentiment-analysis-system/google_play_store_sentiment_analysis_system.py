import numpy as np


def tokenize(text):
    return text.lower().split()


class CountVectorizer:
    """Bag-of-words: learn a vocabulary, then map each review to a vector of
    raw term counts (the input Multinomial Naive Bayes expects)."""

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
                j = self.vocab.get(w)          # ignore words unseen in training
                if j is not None:
                    X[i, j] += 1.0
        return X


class MultinomialNB:
    """Multinomial Naive Bayes from scratch. For each class we estimate a word
    distribution with Laplace (add-one) smoothing, then score a review by the
    log-prior plus the sum of log word-likelihoods and take the argmax."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, v = X.shape
        self.log_prior = np.zeros(len(self.classes))
        self.log_lik = np.zeros((len(self.classes), v))
        for k, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[k] = np.log(len(Xc) / n)
            counts = Xc.sum(axis=0) + self.alpha       # per-word counts + smoothing
            self.log_lik[k] = np.log(counts / counts.sum())
        return self

    def predict_log_proba(self, X):
        # log-joint per class; argmax equals argmax of the posterior.
        return self.log_prior + X @ self.log_lik.T

    def predict(self, X):
        return self.classes[np.argmax(self.predict_log_proba(X), axis=1)]


def make_reviews(n=1200, seed=0):
    # Synthetic Play Store reviews. Each sentiment class draws mostly from its
    # own opinion words plus shared app filler, so the planted signal is real
    # but overlapping (words leak across classes) -- a recoverable structure.
    rng = np.random.RandomState(seed)
    sentiment = {
        0: ["crash", "bug", "terrible", "worst", "broken", "freezes",
            "useless", "annoying", "laggy", "refund", "awful", "hate"],       # negative
        1: ["okay", "average", "fine", "decent", "meh", "ordinary",
            "mediocre", "alright", "passable", "so-so"],                        # neutral
        2: ["love", "great", "excellent", "amazing", "smooth", "best",
            "perfect", "fast", "beautiful", "helpful", "awesome", "fantastic"],# positive
    }
    shared = ["app", "update", "phone", "install", "screen", "version",
              "game", "ads", "feature", "the", "this", "it", "really", "please"]
    names = list(sentiment.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        length = rng.randint(8, 16)
        n_op = max(3, int(length * 0.55))              # ~55% opinion, rest filler
        words = list(rng.choice(sentiment[c], n_op)) + \
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

    docs, y = make_reviews(n=1200, seed=0)
    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

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
    f1 = macro_f1(yte, pred, clf.classes)

    # Majority-class baseline: always predict the most common training class.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, clf.classes)

    print("Reviews: %d   Train: %d   Test: %d   Classes: %d"
          % (len(docs), len(tr), len(te), len(labels)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 60)
    print("Naive Bayes    accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 60)
    samples = ["the app crashes and freezes worst update ever",
               "decent app but really average meh so-so ordinary",
               "love this app amazing smooth fast best game ever",
               "useless buggy laggy please refund terrible ads"]
    for text in samples:
        c = clf.predict(vec.transform([text]))[0]
        print("  '%s...' -> %s" % (text[:40], labels[c]))
    print("-" * 60)
    print("Naive Bayes beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
