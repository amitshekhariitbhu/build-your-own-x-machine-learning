import numpy as np


def tokenize(text):
    return text.lower().split()


# Small hand-built sarcasm lexicons. The tell of sarcasm is CONTRAST:
# positive sentiment words aimed at an unpleasant situation ("love mondays").
POSITIVE = {"love", "great", "amazing", "wonderful", "fantastic", "perfect",
            "awesome", "brilliant", "delightful", "thrilled", "adore", "best"}
NEGATIVE = {"hate", "terrible", "awful", "worst", "boring", "annoying",
            "miserable", "dreadful", "horrible", "ugh"}
UNPLEASANT = {"monday", "traffic", "meeting", "homework", "taxes", "dentist",
              "queue", "delay", "chores", "rain", "bill", "overtime"}
MARKERS = {"oh", "wow", "yeah", "sure", "totally", "obviously", "clearly"}


class Featurizer:
    """Bag-of-words counts + engineered sarcasm cues, from scratch.

    Extra columns encode the contrast signal a plain BoW model can miss:
      contrast   = positive word AND unpleasant-situation word co-occur
      pos/neg/unp/mark = lexicon hit counts
    Columns are standardized with train statistics so gradient descent is stable.
    """

    def _cues(self, toks):
        p = sum(t in POSITIVE for t in toks)
        n = sum(t in NEGATIVE for t in toks)
        u = sum(t in UNPLEASANT for t in toks)
        m = sum(t in MARKERS for t in toks)
        contrast = 1.0 if (p > 0 and u > 0) else 0.0
        return [contrast, p, n, u, m]

    def fit(self, docs):
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        X = self._raw(docs)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        return self

    def _raw(self, docs):
        V = len(self.vocab)
        X = np.zeros((len(docs), V + 5))
        for i, d in enumerate(docs):
            toks = tokenize(d)
            for w in toks:
                j = self.vocab.get(w)          # ignore unseen words
                if j is not None:
                    X[i, j] += 1.0
            X[i, V:] = self._cues(toks)
        return X

    def transform(self, docs):
        return (self._raw(docs) - self.mu) / self.sd


class LogisticRegression:
    """Binary logistic regression via full-batch gradient descent, from scratch.

    p = sigmoid(Xw + b);  loss = BCE + (l2/2)||w||^2
    grad_w = X^T (p - y)/n + l2*w ;  grad_b = mean(p - y)
    """

    def __init__(self, lr=0.5, epochs=400, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))
            err = p - y
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def make_corpus(n=1600, seed=0):
    # Synthetic binary corpus: 1=sarcastic, 0=sincere (positive OR negative).
    # Sincere-positive : positive words about PLEASANT things.
    # Sincere-negative : negative words about UNPLEASANT things.
    # Sarcastic        : positive words about UNPLEASANT things (+ some markers).
    # Overlap is planted: markers appear in only ~60% of sarcasm, and sincere
    # text occasionally borrows the other side's vocabulary -> non-separable.
    rng = np.random.RandomState(seed)
    pleasant = ["vacation", "beach", "party", "gift", "sunshine", "friends",
                "holiday", "picnic", "wedding", "puppy"]
    fill = ["the", "a", "is", "this", "my", "today", "again", "so", "and",
            "it", "was", "really", "what", "day"]
    pos, neg = list(POSITIVE), list(NEGATIVE)
    unp, mrk = list(UNPLEASANT), list(MARKERS)

    docs, labels = [], []
    for _ in range(n):
        r = rng.rand()
        words = list(rng.choice(fill, rng.randint(3, 6)))
        if r < 0.40:                                   # 40% sarcastic
            words += list(rng.choice(pos, rng.randint(1, 3)))
            words += list(rng.choice(unp, rng.randint(1, 3)))
            if rng.rand() < 0.60:                       # only some carry markers
                words += list(rng.choice(mrk, rng.randint(1, 3)))
            lab = 1
        elif r < 0.70:                                 # 30% sincere-positive
            words += list(rng.choice(pos, rng.randint(1, 3)))
            words += list(rng.choice(pleasant, rng.randint(1, 3)))
            if rng.rand() < 0.15:                       # bleed: mild noise
                words += [rng.choice(unp)]
            lab = 0
        else:                                          # 30% sincere-negative
            words += list(rng.choice(neg, rng.randint(1, 3)))
            words += list(rng.choice(unp, rng.randint(1, 3)))
            if rng.rand() < 0.15:                       # bleed: mild noise
                words += [rng.choice(pos)]
            lab = 0
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(lab)
    return docs, np.array(labels)


def prf1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)
    docs, y = make_corpus(n=1600, seed=0)

    # Held-out split: train on 70%, test on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    tr_docs = [docs[i] for i in tr]
    te_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    feat = Featurizer().fit(tr_docs)
    Xtr, Xte = feat.transform(tr_docs), feat.transform(te_docs)
    clf = LogisticRegression().fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    prec, rec, f1 = prf1(yte, pred)

    # Majority-class baseline: always predict the most common training label.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    _, _, base_f1 = prf1(yte, base_pred)

    print("Docs: %d   Train: %d   Test: %d" % (len(docs), len(tr), len(te)))
    print("Test sarcastic=%d  sincere=%d   Vocab: %d"
          % (np.sum(yte == 1), np.sum(yte == 0), len(feat.vocab)))
    print("-" * 60)
    print("Logistic Reg   acc: %.4f  P: %.3f  R: %.3f  F1: %.4f"
          % (acc, prec, rec, f1))
    print("Majority base  acc: %.4f  P: %.3f  R: %.3f  F1: %.4f"
          % (base_acc, 0.0, 0.0, base_f1))
    print("-" * 60)
    samples = ["oh great another monday meeting so love this",
               "beach vacation with friends what a wonderful day",
               "traffic again this is the worst awful commute"]
    for text in samples:
        lab = clf.predict(feat.transform([text]))[0]
        print("  [%-9s] %s" % ("sarcastic" if lab else "sincere", text[:42]))
    print("-" * 60)
    print("Detector beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
