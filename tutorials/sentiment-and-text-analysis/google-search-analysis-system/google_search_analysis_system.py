import numpy as np


def tokenize(text):
    return text.lower().split()


class TfidfVectorizer:
    """Bag-of-words -> TF-IDF from scratch. Learn a vocabulary and document
    frequencies on the training set, then weight each term by its term
    frequency times log inverse-document-frequency and L2-normalise rows."""

    def fit(self, docs):
        self.vocab = {}
        df = {}
        for d in docs:
            seen = set()
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
                if w not in seen:                       # count each doc once
                    df[w] = df.get(w, 0) + 1
                    seen.add(w)
        n = len(docs)
        self.idf = np.ones(len(self.vocab))
        for w, j in self.vocab.items():
            self.idf[j] = np.log((1 + n) / (1 + df[w])) + 1.0   # smoothed idf
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)                   # skip unseen words
                if j is not None:
                    X[i, j] += 1.0
        X *= self.idf                                   # tf * idf
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm == 0, 1.0, norm)       # L2 normalise


class SoftmaxRegression:
    """Multinomial logistic regression trained with full-batch gradient
    descent. Scores = XW+b, softmax to probabilities, cross-entropy loss,
    and analytic gradients -- no autograd, no library optimiser."""

    def __init__(self, lr=0.5, epochs=300, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def _softmax(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, d = X.shape
        k = len(self.classes)
        Y = np.zeros((n, k))
        Y[np.arange(n), np.searchsorted(self.classes, y)] = 1.0   # one-hot
        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        for _ in range(self.epochs):
            P = self._softmax(X @ self.W + self.b)
            gW = X.T @ (P - Y) / n + self.l2 * self.W
            gb = (P - Y).mean(axis=0)
            self.W -= self.lr * gW
            self.b -= self.lr * gb
        return self

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]


def make_search_logs(n=1600, seed=0):
    # Synthetic Google search queries. Each query has a planted INTENT: users
    # phrase informational / navigational / transactional / commercial searches
    # with characteristic words, mixed with shared connective filler so the
    # classes overlap but stay recoverable.
    rng = np.random.RandomState(seed)
    intent = {
        0: ["how", "what", "why", "guide", "tutorial", "meaning", "define",
            "explain", "examples", "history", "facts", "symptoms"],        # informational
        1: ["login", "homepage", "official", "website", "youtube", "gmail",
            "dashboard", "account", "portal", "sign-in", "app"],           # navigational
        2: ["buy", "cheap", "order", "price", "deal", "discount", "coupon",
            "shipping", "checkout", "sale", "download", "subscribe"],      # transactional
        3: ["best", "top", "review", "vs", "compare", "rating", "worth",
            "alternatives", "recommended", "pros", "cons"],                # commercial
    }
    shared = ["near", "me", "online", "for", "in", "2024", "free", "the",
              "of", "with", "new", "and", "to", "a"]
    names = list(intent.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        length = rng.randint(4, 9)
        n_key = max(2, int(length * 0.6))               # ~60% intent words
        words = list(rng.choice(intent[c], n_key)) + \
            list(rng.choice(shared, length - n_key))
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

    docs, y = make_search_logs(n=1600, seed=0)
    labels = ["INFORMATIONAL", "NAVIGATIONAL", "TRANSACTIONAL", "COMMERCIAL"]

    # Held-out split: train on 70%, evaluate on the remaining 30%.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    tr_docs = [docs[i] for i in tr]
    te_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = TfidfVectorizer().fit(tr_docs)
    Xtr, Xte = vec.transform(tr_docs), vec.transform(te_docs)
    clf = SoftmaxRegression(lr=0.5, epochs=300, l2=1e-3).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, clf.classes)

    # Majority-class baseline: always predict the most common training intent.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, clf.classes)

    print("Queries: %d   Train: %d   Test: %d   Intents: %d"
          % (len(docs), len(tr), len(te), len(labels)))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 62)
    print("Softmax TF-IDF  accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class  accuracy: %.4f   macro-F1: %.4f"
          % (base_acc, base_f1))
    print("-" * 62)
    samples = ["how to define recursion tutorial examples",
               "gmail login official website account portal",
               "buy cheap running shoes online discount coupon",
               "best laptops 2024 review compare top rating"]
    for text in samples:
        c = clf.predict(vec.transform([text]))[0]
        print("  '%s' -> %s" % (text, labels[c]))
    print("-" * 62)
    print("Softmax TF-IDF beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
