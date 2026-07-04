import numpy as np


def tokenize(text):
    return text.lower().split()


class CountVectorizer:
    """Bag-of-words term-count matrix, built from scratch."""

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
    """Multinomial Naive Bayes for multi-class text, from scratch.

    log P(c | doc) = log P(c) + sum_w count(w) * log P(w | c)
    with Laplace (add-alpha) smoothing on the per-class word counts.
    All math is done in log-space to avoid underflow.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_feat = X.shape[1]
        self.log_prior = np.zeros(len(self.classes))
        self.log_likelihood = np.zeros((len(self.classes), n_feat))
        for k, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[k] = np.log(Xc.shape[0] / X.shape[0])
            counts = Xc.sum(axis=0) + self.alpha           # smoothed word counts
            self.log_likelihood[k] = np.log(counts / counts.sum())
        return self

    def predict(self, X):
        # scores[i, k] = log-joint of doc i under class k
        scores = X @ self.log_likelihood.T + self.log_prior
        return self.classes[np.argmax(scores, axis=1)]


def make_ads(n=1200, seed=0):
    # Synthetic social-media ad corpus. Each ad is drawn from one campaign
    # category with its own product vocabulary, mixed with shared marketing
    # filler ("shop", "now", "sale") so the planted category signal is strong
    # but overlapping and noisy, like real ad copy.
    rng = np.random.RandomState(seed)
    categories = {
        "Fashion":  ["dress", "shoes", "jacket", "handbag", "denim", "sneakers",
                     "outfit", "boots", "wardrobe", "style", "trendy", "leather"],
        "Tech":     ["laptop", "phone", "wireless", "gadget", "battery", "camera",
                     "smart", "headphones", "charger", "screen", "processor", "5g"],
        "Food":     ["pizza", "burger", "delivery", "tasty", "recipe", "coffee",
                     "snack", "organic", "meal", "fresh", "dessert", "vegan"],
        "Travel":   ["flight", "hotel", "vacation", "beach", "resort", "booking",
                     "getaway", "cruise", "passport", "itinerary", "tour", "island"],
        "Finance":  ["loan", "invest", "savings", "crypto", "insurance", "credit",
                     "budget", "wealth", "mortgage", "trading", "returns", "cashback"],
    }
    shared = ["shop", "now", "sale", "off", "deal", "today", "new", "best",
              "free", "limited", "click", "get", "the", "your", "our", "up"]
    names = list(categories.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.randint(len(names))
        core = categories[names[c]]
        length = rng.randint(8, 16)
        n_core = int(length * 0.55)                        # 55% topical, 45% filler
        words = list(rng.choice(core, n_core)) + \
            list(rng.choice(shared, length - n_core))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(c)
    return docs, np.array(labels), names


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return np.mean(f1s)


if __name__ == "__main__":
    np.random.seed(0)

    docs, y, names = make_ads(n=1200, seed=0)

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
    f1 = macro_f1(yte, pred, len(names))

    # Majority-class baseline: always predict the most common training label.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, len(names))

    print("Ads: %d   Train: %d   Test: %d   Classes: %d"
          % (len(docs), len(tr), len(te), len(names)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 62)
    print("Multinomial NB   accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class   accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 62)
    samples = ["limited sale off leather jacket sneakers denim shop now",
               "new wireless headphones smart phone charger battery deal today",
               "best crypto invest savings loan returns cashback free today",
               "vacation flight hotel resort beach booking getaway deal now"]
    for text in samples:
        p = clf.predict(vec.transform([text]))[0]
        print("  [%-8s] '%s'" % (names[p], text[:44]))
    print("-" * 62)
    print("Naive Bayes beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
