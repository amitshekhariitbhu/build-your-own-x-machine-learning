import numpy as np

# Financial Budget Analysis System
# ---------------------------------
# Auto-categorize free-text bank/card transaction descriptions into budget
# categories, then roll predicted spend up against per-category limits to flag
# overspending. The classifier is a multinomial Naive Bayes built from scratch
# on bag-of-words counts; the budget report is plain aggregation.


def tokenize(text):
    return text.lower().split()


class CountVectorizer:
    """Bag-of-words term-count matrix, from scratch."""

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
    """Multinomial Naive Bayes with Laplace smoothing.

    log P(c | doc) = log P(c) + sum_w count_w * log P(w | c)
    P(w | c) = (count of w in class c + alpha) / (total tokens in c + alpha*V)
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, v = X.shape
        self.log_prior = np.zeros(len(self.classes))
        self.log_lik = np.zeros((len(self.classes), v))     # class x word
        for k, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[k] = np.log(len(Xc) / n)
            counts = Xc.sum(axis=0) + self.alpha             # smoothed word counts
            self.log_lik[k] = np.log(counts / counts.sum())
        return self

    def predict_log_proba(self, X):
        return X @ self.log_lik.T + self.log_prior           # (n_docs x n_classes)

    def predict(self, X):
        return self.classes[np.argmax(self.predict_log_proba(X), axis=1)]


# Category templates: distinctive keywords per budget category plus shared
# filler tokens (dates, generic bank noise) so the planted signal is strong
# but realistically overlapping.
CATEGORIES = {
    "Groceries":     ["supermarket", "grocery", "market", "produce", "bakery",
                      "deli", "walmart", "costco", "aldi", "milk", "eggs"],
    "Dining":        ["restaurant", "cafe", "coffee", "pizza", "burger", "diner",
                      "starbucks", "bar", "grill", "takeout", "bistro"],
    "Transport":     ["uber", "taxi", "fuel", "gas", "parking", "metro", "transit",
                      "shell", "toll", "lyft", "station"],
    "Utilities":     ["electric", "water", "internet", "phone", "utility", "cable",
                      "gas", "power", "broadband", "bill", "energy"],
    "Entertainment": ["netflix", "cinema", "movie", "spotify", "game", "concert",
                      "theater", "steam", "hulu", "ticket", "arcade"],
}
SHARED = ["payment", "purchase", "card", "pos", "debit", "txn", "ref", "online",
          "store", "monthly", "auto", "recurring"]

# Planted per-category budget (dollars) for the held-out period. Essentials are
# comfortably funded; the discretionary categories are set tight so the report
# surfaces realistic overspending on Dining and Entertainment.
BUDGET = {"Groceries": 5500, "Dining": 1800, "Transport": 1300,
          "Utilities": 5500, "Entertainment": 400}


def make_transactions(n=1200, seed=0):
    """Synthetic transactions: a description (text) + an amount + true category."""
    rng = np.random.RandomState(seed)
    cats = list(CATEGORIES.keys())
    all_core = [w for c in cats for w in CATEGORIES[c]]
    # Uneven prevalence so a majority baseline is non-trivial.
    probs = np.array([0.32, 0.24, 0.18, 0.16, 0.10])
    docs, amounts, labels = [], [], []
    for _ in range(n):
        k = rng.choice(len(cats), p=probs)
        cat = cats[k]
        core = CATEGORIES[cat]
        length = rng.randint(4, 8)
        n_core = max(2, int(length * 0.55))                  # ~55% topical, rest filler
        words = list(rng.choice(core, n_core)) + \
            list(rng.choice(SHARED, length - n_core))
        if rng.rand() < 0.20:                                # 20% ambiguous merchants:
            words.append(rng.choice(all_core))               # a stray off-category token
        rng.shuffle(words)
        docs.append(" ".join(words))
        # Category-specific spend distribution (positive, lognormal-ish).
        base = {"Groceries": 45, "Dining": 22, "Transport": 18,
                "Utilities": 70, "Entertainment": 15}[cat]
        amounts.append(round(base * np.exp(rng.normal(0, 0.4)), 2))
        labels.append(k)
    return docs, np.array(amounts), np.array(labels), cats


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

    docs, amounts, y, cats = make_transactions(n=1200, seed=0)

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
    f1 = macro_f1(yte, pred, len(cats))

    # Majority-class baseline: always predict the most common training category.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, len(cats))

    print("Transactions: %d   Train: %d   Test: %d   Categories: %d"
          % (len(docs), len(tr), len(te), len(cats)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 62)
    print("Naive Bayes   accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority base accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("NB beats majority baseline: %s" % (acc > base_acc and f1 > base_f1))
    print("-" * 62)

    # Budget analysis: categorize the held-out (unlabeled-in-practice) spend and
    # compare predicted totals against planted monthly budgets.
    print("Budget report (predicted spend vs monthly limit):")
    te_amounts = amounts[te]
    for c, cat in enumerate(cats):
        spent = te_amounts[pred == c].sum()
        limit = BUDGET[cat]
        status = "OVER " if spent > limit else "ok   "
        bar = "#" * int(min(spent / limit, 1.5) * 20)
        print("  %-14s $%8.2f / $%6d  [%s] %s"
              % (cat, spent, limit, status, bar))
    print("-" * 62)
    samples = ["starbucks coffee card payment",
               "shell fuel gas station debit",
               "netflix monthly recurring online",
               "walmart supermarket grocery purchase"]
    for text in samples:
        c = clf.predict(vec.transform([text]))[0]
        print("  [%-13s] '%s'" % (cats[c], text))
