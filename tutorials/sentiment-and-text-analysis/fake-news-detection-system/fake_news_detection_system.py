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
    """Multinomial Naive Bayes with Laplace smoothing, from scratch.

    log P(y|doc) = log P(y) + sum_w count(w) * log P(w|y)
    P(w|y) is estimated from per-class word totals with add-alpha smoothing,
    and everything is done in log space to avoid float underflow.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_feats = X.shape[1]
        self.log_prior = np.zeros(len(self.classes))
        self.log_likelihood = np.zeros((len(self.classes), n_feats))
        for k, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[k] = np.log(Xc.shape[0] / X.shape[0])
            counts = Xc.sum(axis=0) + self.alpha           # smoothed word counts
            self.log_likelihood[k] = np.log(counts / counts.sum())
        return self

    def predict_log_proba(self, X):
        # scores[i, k] = log P(class_k) + sum_w count * log P(w|class_k)
        return self.log_prior + X @ self.log_likelihood.T

    def predict(self, X):
        return self.classes[self.predict_log_proba(X).argmax(axis=1)]


def make_news(n=1000, seed=0):
    # Synthetic headline corpus. Fake items draw mostly from a sensational /
    # clickbait vocabulary, real items from a measured / factual vocabulary;
    # both share neutral filler so the planted signal is strong but noisy.
    rng = np.random.RandomState(seed)
    fake_words = ["shocking", "secret", "miracle", "exposed", "hoax", "insane",
                  "unbelievable", "cover-up", "banned", "cure", "conspiracy",
                  "outrage", "bombshell", "scam", "leaked", "viral", "aliens",
                  "instantly", "hate", "destroy", "wow", "must-see"]
    real_words = ["report", "official", "study", "committee", "economy",
                  "policy", "senate", "researchers", "announced", "quarterly",
                  "budget", "election", "minister", "data", "analysis",
                  "statement", "market", "reform", "agency", "survey"]
    shared = ["the", "a", "of", "to", "in", "on", "for", "and", "with",
              "new", "after", "over", "says", "as", "this", "year"]

    docs, labels = [], []
    for _ in range(n):
        fake = rng.rand() < 0.45                            # 45% fake prevalence
        core = fake_words if fake else real_words
        length = rng.randint(8, 16)
        n_core = int(length * 0.5)                          # 50% topical, 50% filler
        words = list(rng.choice(core, n_core)) + \
            list(rng.choice(shared, length - n_core))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(1 if fake else 0)                     # 1 = fake, 0 = real
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

    docs, y = make_news(n=1000, seed=0)

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
    prec, rec, f1 = prf(yte, pred)

    # Majority-class baseline: always predict the most common training label.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    _, _, base_f1 = prf(yte, base_pred)

    print("Headlines: %d   Train: %d   Test: %d   Fake rate: %.2f"
          % (len(docs), len(tr), len(te), y.mean()))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 62)
    print("NaiveBayes + BoW  accuracy: %.4f  precision: %.4f  recall: %.4f  F1: %.4f"
          % (acc, prec, rec, f1))
    print("Majority class    accuracy: %.4f  F1: %.4f" % (base_acc, base_f1))
    print("-" * 62)
    samples = ["shocking secret cure exposed the miracle banned hoax viral",
               "senate committee announced new budget policy after study report",
               "unbelievable leaked bombshell conspiracy aliens destroy instantly wow",
               "official quarterly market data analysis says economy reform new"]
    for text in samples:
        lp = clf.predict_log_proba(vec.transform([text]))[0]
        p = np.exp(lp - lp.max())
        p = p / p.sum()
        tag = "FAKE" if p[1] >= 0.5 else "REAL"
        print("  [%s p=%.2f] '%s'" % (tag, p[1], text[:44]))
    print("-" * 62)
    print("NaiveBayes beats majority baseline: %s" % (acc > base_acc and f1 > base_f1))
