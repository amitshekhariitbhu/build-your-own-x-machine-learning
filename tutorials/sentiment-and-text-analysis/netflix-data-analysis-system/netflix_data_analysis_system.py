import numpy as np


def tokenize(text):
    return text.lower().split()


class CountVectorizer:
    """Bag-of-words from scratch: learn a vocabulary from the training
    descriptions, then turn every description into a raw word-count vector."""

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
                j = self.vocab.get(w)            # ignore unseen words
                if j is not None:
                    X[i, j] += 1.0
        return X


class MultinomialNB:
    """Multinomial Naive Bayes trained by counting. For each genre we learn a
    word-probability table with Laplace smoothing, then score a new title as
    log P(genre) + sum of log P(word | genre) and pick the largest."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, d = X.shape
        k = len(self.classes)
        self.log_prior = np.zeros(k)
        self.log_prob = np.zeros((k, d))          # log P(word | class)
        for ci, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[ci] = np.log(len(Xc) / n)
            counts = Xc.sum(axis=0) + self.alpha  # Laplace-smoothed counts
            self.log_prob[ci] = np.log(counts / counts.sum())
        return self

    def predict(self, X):
        scores = X @ self.log_prob.T + self.log_prior   # log joint per class
        return self.classes[np.argmax(scores, axis=1)]


def make_catalog(n=1200, seed=0):
    # Synthetic Netflix catalog. Each title's description blends generic
    # catalog vocabulary (no genre signal) with words drawn from its own genre
    # lexicon, so the planted genre structure is genuinely recoverable.
    rng = np.random.RandomState(seed)
    genre = {
        0: ["laugh", "comedy", "hilarious", "sitcom", "prank", "witty",
            "goofy", "standup", "awkward", "quirky", "banter", "parody"],        # Comedy
        1: ["haunted", "ghost", "terror", "slasher", "cursed", "nightmare",
            "demon", "scream", "possessed", "creepy", "supernatural", "grave"],  # Horror
        2: ["nature", "history", "investigation", "footage", "interview",
            "archive", "science", "wildlife", "biography", "factual",
            "expedition", "climate"],                                            # Documentary
        3: ["love", "romance", "wedding", "heartbreak", "kiss", "affair",
            "soulmate", "passion", "longing", "courtship", "tender", "devotion"],# Romance
        4: ["explosion", "chase", "heist", "gunfight", "mission", "assassin",
            "combat", "escape", "showdown", "rescue", "sabotage", "pursuit"],    # Action
    }
    catalog = ["series", "season", "episode", "story", "cast", "character",
               "world", "journey", "night", "city", "family", "friend",
               "secret", "dream", "stream", "release", "trailer", "premiere",
               "star", "director", "screen", "runtime", "plot", "scene"]
    names = list(genre.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        length = rng.randint(12, 22)
        n_genre = max(4, int(length * 0.45))       # ~45% genre words, rest generic
        words = list(rng.choice(genre[c], n_genre)) + \
            list(rng.choice(catalog, length - n_genre))
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

    docs, y = make_catalog(n=1200, seed=0)
    label_names = ["COMEDY", "HORROR", "DOCUMENTARY", "ROMANCE", "ACTION"]

    # Held-out split: train on 70% of the catalog, evaluate on unseen 30%.
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

    # Majority-class baseline: always guess the most common training genre.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, clf.classes)

    print("Titles: %d   Train: %d   Test: %d   Genres: %d"
          % (len(docs), len(tr), len(te), len(label_names)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 62)
    print("Bag-of-words + Naive Bayes accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority genre        baseline  acc: %.4f   macro-F1: %.4f"
          % (base_acc, base_f1))
    print("-" * 62)
    samples = [
        "haunted ghost cursed nightmare demon night city secret scene",
        "love wedding heartbreak soulmate passion story family dream",
        "explosion chase heist mission assassin escape city night plot",
        "nature wildlife interview archive science climate world journey",
    ]
    for text in samples:
        c = clf.predict(vec.transform([text]))[0]
        print("  '%s...' -> %s" % (text[:42], label_names[c]))
    print("-" * 62)
    print("Naive Bayes beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
