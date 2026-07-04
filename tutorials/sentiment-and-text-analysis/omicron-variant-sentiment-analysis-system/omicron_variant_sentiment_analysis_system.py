import numpy as np


def tokenize(text):
    return text.lower().split()


class CountVectorizer:
    """Bag-of-words counts from scratch: builds a vocabulary from the training
    corpus, then maps each document to a vector of raw term frequencies."""

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


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes from scratch. Learns log class priors and
    Laplace-smoothed log word likelihoods, then scores a document as the class
    maximising log P(class) + sum(count * log P(word|class))."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha                     # Laplace smoothing strength

    def fit(self, X, y):
        n, d = X.shape
        self.classes = np.unique(y)
        k = len(self.classes)
        self.log_prior = np.zeros(k)
        self.log_like = np.zeros((k, d))       # log P(word | class)
        for ci, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[ci] = np.log(len(Xc) / n)
            counts = Xc.sum(0) + self.alpha    # smoothed word counts
            self.log_like[ci] = np.log(counts / counts.sum())
        return self

    def predict_log_proba(self, X):
        # Joint log score per class; normalised into a proper log-posterior.
        scores = X @ self.log_like.T + self.log_prior
        m = scores.max(axis=1, keepdims=True)
        logZ = m + np.log(np.exp(scores - m).sum(axis=1, keepdims=True))
        return scores - logZ

    def predict(self, X):
        return self.classes[np.argmax(self.predict_log_proba(X), axis=1)]


def make_omicron_posts(n=1200, seed=0):
    # Synthetic Omicron-variant social posts. Each sentiment class samples mostly
    # from its own opinion words plus shared pandemic chatter, so the planted
    # sentiment signal is recoverable from word counts alone.
    rng = np.random.RandomState(seed)
    sentiment = {
        0: ["scared", "deadly", "surge", "hospital", "overwhelmed", "lockdown",
            "worried", "spike", "dangerous", "fear", "crisis", "panic"],       # negative
        1: ["reported", "variant", "update", "cases", "study", "data", "who",
            "briefing", "testing", "guidance", "report", "monitoring"],         # neutral
        2: ["mild", "recovering", "hopeful", "vaccine", "boosters", "relief",
            "protected", "recovery", "optimistic", "safe", "declining", "good"],  # positive
    }
    shared = ["omicron", "covid", "the", "a", "this", "new", "people", "today",
              "about", "more", "with", "just"]
    names = list(sentiment.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        length = rng.randint(9, 16)
        n_topic = max(3, int(length * 0.55))         # 55% opinion, 45% chatter
        words = list(rng.choice(sentiment[c], n_topic)) + \
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

    docs, y = make_omicron_posts(n=1200, seed=0)
    labels_txt = ["negative", "neutral", "positive"]

    # Held-out split: train on 70%, test on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    tr_docs = [docs[i] for i in tr]
    te_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = CountVectorizer().fit(tr_docs)
    Xtr, Xte = vec.transform(tr_docs), vec.transform(te_docs)
    clf = MultinomialNaiveBayes(alpha=1.0).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, clf.classes)

    # Majority-class baseline: always predict the most common training sentiment.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, clf.classes)

    print("Posts: %d   Train: %d   Test: %d   Classes: %d"
          % (len(docs), len(tr), len(te), len(labels_txt)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 60)
    print("Naive Bayes    accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 60)
    samples = ["omicron surge overwhelmed hospital scared lockdown crisis",
               "omicron variant cases reported who briefing data update",
               "omicron mild recovering hopeful boosters relief declining"]
    for text in samples:
        c = clf.predict(vec.transform([text]))[0]
        print("  '%s...' -> %s" % (text[:40], labels_txt[c]))
    print("-" * 60)
    print("Naive Bayes beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
