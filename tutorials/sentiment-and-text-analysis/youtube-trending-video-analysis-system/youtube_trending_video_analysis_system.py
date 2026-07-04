import numpy as np


def tokenize(text):
    return text.lower().split()


class MultinomialNB:
    """Multinomial Naive Bayes from scratch. Learn a per-class word-count
    distribution with Laplace smoothing, then score a new video's title+tags
    by summing log P(word|class) plus the log class prior. The class with the
    highest posterior log-probability wins."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha                            # Laplace smoothing

    def fit(self, docs, y):
        self.classes = np.unique(y)
        self.vocab = {}
        for d in docs:                                # build vocabulary
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        v = len(self.vocab)
        k = len(self.classes)
        counts = np.zeros((k, v))                      # word counts per class
        class_docs = np.zeros(k)
        for d, label in zip(docs, y):
            ci = np.searchsorted(self.classes, label)
            class_docs[ci] += 1
            for w in tokenize(d):
                counts[ci, self.vocab[w]] += 1.0
        counts += self.alpha                           # smooth
        self.log_likelihood = np.log(counts / counts.sum(axis=1, keepdims=True))
        self.log_prior = np.log(class_docs / class_docs.sum())
        return self

    def _features(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)                  # ignore unseen words
                if j is not None:
                    X[i, j] += 1.0
        return X

    def predict(self, docs):
        X = self._features(docs)
        scores = X @ self.log_likelihood.T + self.log_prior   # log posterior
        return self.classes[np.argmax(scores, axis=1)]


def make_trending_videos(n=1200, seed=0):
    # Synthetic YouTube trending video metadata. Each video's title+tags mix
    # generic platform chatter (shared, no signal) with words drawn from its
    # true category's lexicon, so the planted category structure is recoverable.
    rng = np.random.RandomState(seed)
    categories = {
        0: ["song", "album", "official", "remix", "lyrics", "beat", "concert",
            "live", "cover", "audio", "single", "melody", "vocals", "chorus"],   # Music
        1: ["gameplay", "stream", "boss", "level", "speedrun", "loot", "raid",
            "fps", "update", "patch", "montage", "clutch", "respawn", "quest"],   # Gaming
        2: ["breaking", "report", "election", "policy", "market", "coverage",
            "headline", "briefing", "protest", "verdict", "senate", "economy"],   # News
        3: ["prank", "sketch", "hilarious", "standup", "meme", "parody", "joke",
            "reaction", "roast", "blooper", "spoof", "impression", "punchline"],  # Comedy
        4: ["unboxing", "review", "benchmark", "gadget", "specs", "teardown",
            "firmware", "processor", "battery", "camera", "flagship", "gpu"],     # Tech
    }
    common = ["video", "trending", "watch", "subscribe", "channel", "views",
              "upload", "viral", "today", "new", "episode", "part", "full",
              "best", "top", "clip", "highlights", "official", "share"]
    names = list(categories.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        length = rng.randint(9, 16)
        n_cat = max(3, int(length * 0.5))              # ~50% category, rest common
        words = list(rng.choice(categories[c], n_cat)) + \
            list(rng.choice(common, length - n_cat))
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

    docs, y = make_trending_videos(n=1200, seed=0)
    label_names = ["MUSIC", "GAMING", "NEWS", "COMEDY", "TECH"]

    # Held-out split: train on 70%, evaluate on the unseen 30%.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    tr_docs = [docs[i] for i in tr]
    te_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    clf = MultinomialNB(alpha=1.0).fit(tr_docs, ytr)
    pred = clf.predict(te_docs)

    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, clf.classes)

    # Majority-class baseline: always guess the most common training category.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, clf.classes)

    print("Trending videos: %d   Train: %d   Test: %d   Categories: %d"
          % (len(docs), len(tr), len(te), len(label_names)))
    print("Vocabulary size: %d" % len(clf.vocab))
    print("-" * 64)
    print("Naive Bayes   accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority base accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 64)
    samples = [
        "official song remix concert live audio subscribe views viral",
        "gameplay boss speedrun clutch montage raid update trending clip",
        "breaking election coverage senate economy report headline today",
        "prank hilarious sketch reaction roast meme viral watch channel",
        "unboxing review benchmark gpu specs teardown flagship battery new",
    ]
    for text in samples:
        c = clf.predict([text])[0]
        print("  '%s...' -> %s" % (text[:44], label_names[c]))
    print("-" * 64)
    print("Naive Bayes beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
