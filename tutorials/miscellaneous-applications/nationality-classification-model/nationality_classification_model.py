import numpy as np

# Classify a surname into its nationality from character shape alone.
# Idea: each nationality leaves a fingerprint in its character n-grams
# (which letters follow which, and how names end). We turn a name into a
# bag of padded character bigrams and train a softmax (multinomial logistic
# regression) classifier from scratch with gradient descent.

# ---- Synthetic nationalities: distinct consonant/vowel sets + suffixes -------
# Overlapping vowels keep it non-trivial; consonants and endings carry signal.
PROFILES = {
    "Nordish": dict(con="bkrsvthnld", vow="aeiou",  suf=["sen", "sson", "vik", "dal"]),
    "Slavic":  dict(con="vkrszchbnj", vow="aeiou",  suf=["ov", "ski", "enko", "ich"]),
    "Latino":  dict(con="rlmnsgztcd", vow="aeio",   suf=["ez", "os", "ini", "aro"]),
    "Nihon":   dict(con="kstnmhyrwg", vow="aeiou",  suf=["ko", "ta", "shi", "moto"]),
}
LABELS = list(PROFILES)


def make_name(profile):
    """Build a name from consonant-vowel syllables plus a nationality suffix."""
    n_syl = np.random.randint(2, 4)
    s = "".join(np.random.choice(list(profile["con"])) +
                np.random.choice(list(profile["vow"])) for _ in range(n_syl))
    return s + profile["suf"][np.random.randint(len(profile["suf"]))]


def bigrams(name):
    p = "^" + name + "$"                       # boundary markers capture endings
    return [p[i:i + 2] for i in range(len(p) - 1)]


class NationalityClassifier:
    """Char-bigram bag-of-features + softmax regression, trained by GD."""

    def __init__(self, lr=0.5, l2=1e-3, epochs=300):
        self.lr, self.l2, self.epochs = lr, l2, epochs
        self.vocab = {}

    def _featurize(self, names):
        X = np.zeros((len(names), len(self.vocab)))
        for i, nm in enumerate(names):
            for g in bigrams(nm):
                j = self.vocab.get(g)
                if j is not None:
                    X[i, j] += 1.0
        return X

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, names, y):
        self.vocab = {g: j for j, g in                # vocab from training only
                      enumerate(sorted({g for nm in names for g in bigrams(nm)}))}
        X = self._featurize(names)
        n, V = X.shape
        C = int(y.max()) + 1
        Y = np.eye(C)[y]                              # one-hot targets
        self.W = np.zeros((V, C))
        self.b = np.zeros(C)
        for _ in range(self.epochs):
            P = self._softmax(X @ self.W + self.b)
            G = (P - Y) / n
            self.W -= self.lr * (X.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(axis=0)
        return self

    def predict(self, names):
        P = self._softmax(self._featurize(names) @ self.W + self.b)
        return P.argmax(axis=1)


def f1_macro(y, yhat, C):
    fs = []
    for c in range(C):
        tp = np.sum((yhat == c) & (y == c))
        fp = np.sum((yhat == c) & (y != c))
        fn = np.sum((yhat != c) & (y == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(fs))


# Example usage
if __name__ == "__main__":
    np.random.seed(0)

    # Planted structure: 250 names per nationality drawn from its own profile.
    names, y = [], []
    for c, lab in enumerate(LABELS):
        for _ in range(250):
            names.append(make_name(PROFILES[lab]))
            y.append(c)
    names, y = np.array(names), np.array(y)

    idx = np.random.permutation(len(y))               # shuffle then 70/30 split
    names, y = names[idx], y[idx]
    cut = int(0.7 * len(y))
    Xtr, ytr = names[:cut], y[:cut]
    Xte, yte = names[cut:], y[cut:]

    clf = NationalityClassifier().fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = float(np.mean(pred == yte))
    f1 = f1_macro(yte, pred, len(LABELS))
    maj = int(np.bincount(ytr).argmax())              # majority-class baseline
    acc_maj = float(np.mean(yte == maj))

    print("Sample predictions (name -> pred / truth):")
    for nm, p, t in list(zip(Xte, pred, yte))[:6]:
        print("  %-12s -> %-8s (%s)" % (nm, LABELS[p], LABELS[t]))
    print()
    print("Nationalities              :", ", ".join(LABELS))
    print("Train / test names         :", len(ytr), "/", len(yte))
    print("Vocabulary (char bigrams)  :", len(clf.vocab))
    print("Majority-class  accuracy   :", round(acc_maj, 3))
    print("Random-guess    accuracy   :", round(1.0 / len(LABELS), 3))
    print("Classifier      accuracy   :", round(acc, 3))
    print("Classifier      macro-F1   :", round(f1, 3))
    print()
    ok = acc > 0.85 and acc > 3 * acc_maj
    print("RESULT:", "PASS - classifier beats majority baseline" if ok else "FAIL")
