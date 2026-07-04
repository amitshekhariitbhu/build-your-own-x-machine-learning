import numpy as np


def tokenize(text):
    return text.lower().split()


class CountVectorizer:
    """Plain bag-of-words: raw term counts, built from scratch."""

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
                j = self.vocab.get(w)              # ignore unseen words
                if j is not None:
                    X[i, j] += 1.0
        return X


class MultinomialNB:
    """Multinomial Naive Bayes with Laplace smoothing, from scratch.

    log P(c)         = log(class prior)
    log P(w | c)     = log((count(w,c) + a) / (total(c) + a*V))
    predict          = argmax_c [ log P(c) + sum_w x_w * log P(w|c) ]
    All done in log-space to avoid underflow on long documents.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, v = X.shape
        self.log_prior = np.zeros(len(self.classes))
        self.log_likelihood = np.zeros((len(self.classes), v))
        for k, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[k] = np.log(Xc.shape[0] / n)
            counts = Xc.sum(axis=0) + self.alpha            # smoothed word counts
            self.log_likelihood[k] = np.log(counts / counts.sum())
        return self

    def predict_log_proba(self, X):
        # joint log-score per class; shape (n_docs, n_classes)
        return X @ self.log_likelihood.T + self.log_prior

    def predict(self, X):
        return self.classes[np.argmax(self.predict_log_proba(X), axis=1)]


def make_tweets(n=1500, seed=0):
    # Synthetic 3-class corpus: 0=neutral, 1=offensive, 2=hate.
    # Each class draws from a topical vocabulary; hate and offensive SHARE an
    # aggressive vocabulary so they overlap (realistic + non-trivial), while
    # hate is distinguished by group-target + dehumanizing markers. Neutral
    # everyday chatter. Placeholder tokens stand in for real slurs by design.
    rng = np.random.RandomState(seed)
    neutral = ["weather", "coffee", "movie", "game", "recipe", "travel",
               "music", "photo", "weekend", "garden", "book", "sunset",
               "coding", "hiking", "puppy", "market", "lunch", "concert"]
    aggressive = ["idiot", "stupid", "trash", "loser", "moron", "shut",
                  "pathetic", "dumb", "clown", "worthless"]          # shared 1 & 2
    hate_markers = ["grp_target", "subhuman", "vermin", "eliminate",
                    "deport", "inferior", "purge", "threat_them"]    # hate only
    shared = ["the", "you", "a", "and", "is", "to", "this", "of",
              "so", "are", "they", "just", "on", "it", "all"]

    docs, labels = [], []
    for _ in range(n):
        r = rng.rand()
        length = rng.randint(10, 20)
        if r < 0.50:                                   # 50% neutral
            core, extra, lab = neutral, [], 0
        elif r < 0.80:                                 # 30% offensive
            core, extra, lab = aggressive, [], 1
        else:                                          # 20% hate
            core, lab = aggressive, 2
            # only ~65% of hate tweets carry explicit markers; the rest read
            # like plain offensive text -> genuine, non-separable overlap
            extra = hate_markers if rng.rand() < 0.65 else []
        n_core = int(length * 0.45)
        n_extra = int(length * 0.20) if extra else 0
        n_fill = length - n_core - n_extra
        words = (list(rng.choice(core, n_core)) +
                 (list(rng.choice(extra, n_extra)) if n_extra else []) +
                 list(rng.choice(shared, n_fill)))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(lab)
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
    classes = np.array([0, 1, 2])
    names = {0: "neutral", 1: "offensive", 2: "hate"}

    docs, y = make_tweets(n=1500, seed=0)

    # Held-out split: train on 70%, test on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Xtr_docs = [docs[i] for i in tr]
    Xte_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = CountVectorizer().fit(Xtr_docs)
    Xtr, Xte = vec.transform(Xtr_docs), vec.transform(Xte_docs)
    clf = MultinomialNB(alpha=1.0).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, classes)

    # Majority-class baseline: always predict most common training label.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, classes)

    print("Tweets: %d   Train: %d   Test: %d" % (len(docs), len(tr), len(te)))
    print("Class distribution (test): " +
          "  ".join("%s=%d" % (names[c], np.sum(yte == c)) for c in classes))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 62)
    print("Multinomial NB   accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class   accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 62)
    samples = ["coffee sunset weekend hiking photo music",
               "you are a stupid pathetic clown loser",
               "grp_target subhuman vermin deport eliminate inferior"]
    for text in samples:
        p = clf.predict(vec.transform([text]))[0]
        print("  [%-9s] '%s'" % (names[p], text[:44]))
    print("-" * 62)
    print("NB beats majority baseline: %s" % (acc > base_acc and f1 > base_f1))
