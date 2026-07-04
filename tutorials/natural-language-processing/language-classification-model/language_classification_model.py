import numpy as np


def char_ngrams(text, nmin=1, nmax=3):
    """Character n-grams (1..3) with boundary spaces -> the fingerprint of a language."""
    t = " " + text.strip() + " "
    grams = []
    for n in range(nmin, nmax + 1):
        for i in range(len(t) - n + 1):
            grams.append(t[i:i + n])
    return grams


class CharNgramVectorizer:
    """Text -> character n-gram count matrix; vocabulary learned from training only."""

    def fit(self, docs):
        self.vocab = {}
        for d in docs:
            for g in char_ngrams(d):
                self.vocab.setdefault(g, len(self.vocab))
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for g in char_ngrams(d):
                j = self.vocab.get(g)
                if j is not None:
                    X[i, j] += 1.0
        return X


class MultinomialNB:
    """Multinomial Naive Bayes: per-class n-gram likelihoods with Laplace smoothing."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, d = X.shape
        self.log_prior = np.zeros(len(self.classes))
        self.log_prob = np.zeros((len(self.classes), d))
        for i, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[i] = np.log(Xc.shape[0] / n)
            counts = Xc.sum(0) + self.alpha            # smoothed n-gram counts
            self.log_prob[i] = np.log(counts / counts.sum())
        return self

    def predict(self, X):
        scores = X @ self.log_prob.T + self.log_prior  # log P(class | text)
        return self.classes[scores.argmax(1)]


def make_languages(n=700, seed=0):
    # Four synthetic languages. Each draws from its own syllable inventory, so its
    # character n-gram distribution is distinctive; a small shared/borrowed pool adds
    # cross-language noise while the planted per-language signal stays recoverable.
    rng = np.random.RandomState(seed)
    syllables = {
        0: ["vo", "la", "ra", "mo", "na", "li", " vel", "aro", "nu", "ma"],   # "Volan"
        1: ["ke", "th", "si", "te", "ki", "she", "tis", "eth", "ne", "si"],   # "Kethi"
        2: ["zu", "qo", "xa", "gu", "oz", "quo", "zag", "xu", "og", "za"],     # "Zunqo"
        3: ["bi", "de", "pi", "fe", "bel", "dib", "pef", "im", "ed", "fi"],    # "Bimmel"
    }
    borrow = ["an", "el", "or", "in", "us"]                                    # shared filler
    names = list(syllables.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        n_words = rng.randint(4, 9)
        words = []
        for _ in range(n_words):
            n_syl = rng.randint(2, 4)
            parts = list(rng.choice(syllables[c], n_syl))
            if rng.rand() < 0.2:                        # 20% chance to borrow a shared chunk
                parts[rng.randint(n_syl)] = rng.choice(borrow)
            words.append("".join(p.strip() for p in parts))
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

    docs, y = make_languages(n=700, seed=0)
    langs = ["Volan", "Kethi", "Zunqo", "Bimmel"]

    # Held-out split: train on 70%, test on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Xtr_docs = [docs[i] for i in tr]
    Xte_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = CharNgramVectorizer().fit(Xtr_docs)
    Xtr, Xte = vec.transform(Xtr_docs), vec.transform(Xte_docs)
    clf = MultinomialNB().fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, clf.classes)

    # Majority-class baseline: always predict the most common training language.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, clf.classes)

    print("Samples: %d   Train: %d   Test: %d   Languages: %d"
          % (len(docs), len(tr), len(te), len(langs)))
    print("Char n-gram vocabulary size: %d" % len(vec.vocab))
    print("-" * 58)
    print("Char n-gram NB  accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class  accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 58)
    for text in [docs[i] for i in te[:4]]:
        c = clf.predict(vec.transform([text]))[0]
        print("  '%s...' -> %s" % (text[:34], langs[c]))
    print("-" * 58)
    print("Char n-gram NB beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
