import numpy as np

# Fake product review detection from scratch.
# Genuine reviews use a diverse, specific vocabulary, varied ratings, and high
# lexical diversity. Fake reviews recycle a tiny pool of promotional words,
# push extreme ratings, and pile on exclamation marks -> low diversity. We
# plant that latent structure, turn each review into hand-built bag-of-words +
# style features, and train a from-scratch logistic-regression detector.

GENUINE_WORDS = "battery screen delivery month broke returned size color fit \
quality material expected arrived cable button charge week comfortable heavy \
light price average okay works fine".split()
FAKE_WORDS = "amazing best perfect love incredible highly recommend excellent \
awesome fantastic must buy wow great".split()
COMMON = "the and it was a product use i this to".split()
VOCAB = COMMON + GENUINE_WORDS + FAKE_WORDS
WORD_IDX = {w: i for i, w in enumerate(VOCAB)}


def make_review(fake):
    # Return (tokens, rating, exclamations) for one synthetic review.
    if fake:
        n = np.random.randint(6, 14)
        pool = list(np.random.choice(FAKE_WORDS, 4, replace=False)) + COMMON[:3]
        tokens = list(np.random.choice(pool, n))          # repetitive promo
        rating = int(np.random.choice([1, 5], p=[0.1, 0.9]))  # extreme
        excl = np.random.randint(2, 6)
    else:
        n = np.random.randint(10, 30)
        tokens = list(np.random.choice(GENUINE_WORDS + COMMON, n))
        rating = np.random.randint(1, 6)                  # varied
        excl = np.random.randint(0, 2)
    return tokens, rating, excl


def featurize(reviews):
    # Bag-of-words term frequencies + 4 hand-built style features per review.
    rows = []
    for tokens, rating, excl in reviews:
        bow = np.zeros(len(VOCAB))
        for t in tokens:
            bow[WORD_IDX[t]] += 1
        bow /= len(tokens)
        diversity = len(set(tokens)) / len(tokens)  # unique / total
        extremity = abs(rating - 3) / 2.0           # 1&5 star -> 1.0
        rows.append(np.concatenate([bow, [len(tokens), diversity,
                                          extremity, excl]]))
    return np.array(rows)


# Logistic regression trained with batch gradient descent.
class LogisticRegression:
    def __init__(self, lr=0.5, epochs=600):
        self.lr, self.epochs = lr, epochs

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            grad = p - y
            self.w -= self.lr * X.T @ grad / len(y)
            self.b -= self.lr * grad.mean()
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)


def auc_score(y, s):
    # Mann-Whitney U form of ROC AUC using average ranks (handles ties).
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos, neg = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


if __name__ == "__main__":
    np.random.seed(0)
    n, frac_fake = 1200, 0.3
    y = (np.random.rand(n) < frac_fake).astype(int)
    reviews = [make_review(bool(f)) for f in y]
    X = featurize(reviews)

    # 70/30 split, then standardize using train statistics only.
    perm = np.random.permutation(n)
    X, y = X[perm], y[perm]
    cut = int(0.7 * n)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    model = LogisticRegression().fit(Xtr, ytr)
    scores = model.predict_proba(Xte)
    pred = (scores >= 0.5).astype(int)

    tp = int(((pred == 1) & (yte == 1)).sum())
    precision = tp / max(1, (pred == 1).sum())
    recall = tp / max(1, (yte == 1).sum())
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    auc = auc_score(yte, scores)

    # Baselines: prevalence of fakes (random classifier precision) and AUC 0.5.
    prevalence = yte.mean()

    print("Held-out reviews   :", len(yte))
    print("Fake prevalence    : {:.3f}  (random-guess precision)".format(prevalence))
    print("Detector precision : {:.3f}".format(precision))
    print("Detector recall    : {:.3f}".format(recall))
    print("Detector F1        : {:.3f}".format(f1))
    print("Detector ROC AUC   : {:.3f}  (random = 0.500)".format(auc))
    print("Beats baseline     :", precision > prevalence + 0.2 and auc > 0.8)
