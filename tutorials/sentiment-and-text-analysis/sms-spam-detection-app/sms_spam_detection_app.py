import numpy as np


def tokenize(text):
    return text.lower().split()


class TfidfVectorizer:
    """Bag-of-words with TF-IDF weighting, built from scratch.

    tf   = raw term counts per message
    idf  = log((1 + N) / (1 + df)) + 1   (smoothed, sklearn-style)
    Rows are L2-normalized so message length does not dominate.
    """

    def fit(self, docs):
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        df = np.zeros(len(self.vocab))
        for d in docs:
            for w in set(tokenize(d)):
                df[self.vocab[w]] += 1.0
        n = len(docs)
        self.idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)          # ignore unseen words
                if j is not None:
                    X[i, j] += 1.0
        X *= self.idf                          # weight counts by IDF
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm == 0, 1.0, norm)


class LogisticRegression:
    """Binary logistic regression trained with full-batch gradient descent."""

    def __init__(self, lr=0.5, epochs=300, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y                                    # dL/dz
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def make_sms(n=1000, seed=0):
    # Synthetic SMS corpus. Spam samples mostly from a promo/scam vocabulary,
    # ham from everyday chatter; both share filler words so the planted spam
    # signal is strong but noisy (overlapping, realistic).
    rng = np.random.RandomState(seed)
    spam_words = ["free", "win", "winner", "cash", "prize", "claim", "urgent",
                  "offer", "click", "congratulations", "txt", "reply", "call",
                  "guaranteed", "cheap", "credit", "loan", "voucher", "bonus"]
    ham_words = ["meeting", "dinner", "home", "later", "tomorrow", "lunch",
                 "project", "coffee", "movie", "tonight", "mom", "school",
                 "run", "tired", "sorry", "miss", "weekend", "study", "game"]
    shared = ["the", "you", "i", "to", "a", "and", "is", "me", "on", "at",
              "your", "for", "we", "u", "ok", "now"]

    docs, labels = [], []
    for _ in range(n):
        spam = rng.rand() < 0.4                             # 40% spam prevalence
        core = spam_words if spam else ham_words
        length = rng.randint(8, 18)
        n_core = int(length * 0.55)                         # 55% topical, 45% filler
        words = list(rng.choice(core, n_core)) + \
            list(rng.choice(shared, length - n_core))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(1 if spam else 0)                     # 1 = spam, 0 = ham
    return docs, np.array(labels)


def prf(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)

    docs, y = make_sms(n=1000, seed=0)

    # Held-out split: train on 70%, test on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Xtr_docs = [docs[i] for i in tr]
    Xte_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = TfidfVectorizer().fit(Xtr_docs)
    Xtr, Xte = vec.transform(Xtr_docs), vec.transform(Xte_docs)
    clf = LogisticRegression(lr=0.5, epochs=300, l2=1e-3).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    prec, rec, f1 = prf(yte, pred)

    # Majority-class baseline: always predict the most common training label.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    _, _, base_f1 = prf(yte, base_pred)

    print("Messages: %d   Train: %d   Test: %d   Spam rate: %.2f"
          % (len(docs), len(tr), len(te), y.mean()))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 60)
    print("LogReg + TF-IDF  accuracy: %.4f  precision: %.4f  recall: %.4f  F1: %.4f"
          % (acc, prec, rec, f1))
    print("Majority class   accuracy: %.4f  F1: %.4f" % (base_acc, base_f1))
    print("-" * 60)
    samples = ["free cash prize claim now urgent click reply to win",
               "are we still on for dinner tonight at home later",
               "congratulations winner guaranteed voucher call now",
               "sorry running late from the meeting see you tomorrow"]
    for text in samples:
        p = clf.predict_proba(vec.transform([text]))[0]
        tag = "SPAM" if p >= 0.5 else "HAM "
        print("  [%s p=%.2f] '%s'" % (tag, p, text[:40]))
    print("-" * 60)
    print("LogReg beats majority baseline: %s" % (acc > base_acc and f1 > base_f1))
