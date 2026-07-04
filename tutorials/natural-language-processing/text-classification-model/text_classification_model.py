import numpy as np


def tokenize(text):
    return text.lower().split()


class CountVectorizer:
    """Bag-of-words: build a vocabulary and turn docs into term-count rows."""

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
                j = self.vocab.get(w)      # ignore unseen words
                if j is not None:
                    X[i, j] += 1.0
        return X


class MultinomialNaiveBayes:
    """Multinomial NB from scratch: class priors + Laplace-smoothed word
    likelihoods, scored in log-space to avoid float underflow."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha                 # additive (Laplace) smoothing

    def fit(self, X, y):
        n, v = X.shape
        self.classes = np.unique(y)
        k = len(self.classes)
        self.log_prior = np.zeros(k)
        self.log_likelihood = np.zeros((k, v))
        for c_idx, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[c_idx] = np.log(len(Xc) / n)
            counts = Xc.sum(0) + self.alpha            # per-word counts + smoothing
            self.log_likelihood[c_idx] = np.log(counts / counts.sum())
        return self

    def predict(self, X):
        # log P(c|doc) proportional to log_prior + sum(count * log P(word|c)).
        scores = self.log_prior[None, :] + X @ self.log_likelihood.T
        return self.classes[np.argmax(scores, axis=1)]


def make_docs(n=900, seed=0):
    # Synthetic corpus: each class samples mostly from its own topic words
    # plus shared filler, so the planted topic signal is recoverable but noisy.
    rng = np.random.RandomState(seed)
    topics = {
        0: ["refund", "shipping", "order", "package", "delivery", "return", "tracking", "box"],
        1: ["bug", "crash", "error", "login", "password", "app", "update", "screen"],
        2: ["price", "invoice", "charge", "billing", "payment", "subscription", "card", "fee"],
        3: ["thanks", "great", "love", "amazing", "recommend", "happy", "excellent", "perfect"],
    }
    shared = ["the", "i", "my", "was", "please", "you", "it", "and", "for", "with"]
    names = list(topics.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        length = rng.randint(10, 20)
        n_topic = int(length * 0.55)               # 55% topical, 45% filler
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

    docs, y = make_docs(n=900, seed=0)
    categories = ["Shipping", "Technical", "Billing", "Praise"]

    # Held-out split: train on 70%, test on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Xtr_docs = [docs[i] for i in tr]
    Xte_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = CountVectorizer().fit(Xtr_docs)
    Xtr, Xte = vec.transform(Xtr_docs), vec.transform(Xte_docs)
    clf = MultinomialNaiveBayes(alpha=1.0).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, clf.classes)

    # Majority-class baseline: always predict the most common training label.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, clf.classes)

    print("Documents: %d   Train: %d   Test: %d   Classes: %d"
          % (len(docs), len(tr), len(te), len(categories)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 60)
    print("Naive Bayes    accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 60)
    samples = ["my order package delivery tracking was late please",
               "the app crash error login password bug update",
               "invoice charge billing payment subscription fee card",
               "thanks great love amazing recommend excellent perfect"]
    for text in samples:
        c = clf.predict(vec.transform([text]))[0]
        print("  '%s...' -> %s" % (text[:38], categories[c]))
    print("-" * 60)
    print("Naive Bayes beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
