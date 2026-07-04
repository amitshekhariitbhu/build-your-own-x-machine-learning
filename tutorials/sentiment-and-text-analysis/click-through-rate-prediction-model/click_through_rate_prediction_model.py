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
                j = self.vocab.get(w)                  # ignore unseen words
                if j is not None:
                    X[i, j] += 1.0
        return X


class LogisticRegression:
    """L2-regularized logistic regression via full-batch gradient descent.

    p(click | x) = sigmoid(x . w + b). Trained by minimizing the mean
    binary cross-entropy; the gradient of BCE w.r.t. the logit is simply
    (p - y), which keeps the update rule compact and fully vectorized.
    """

    def __init__(self, lr=0.5, epochs=300, l2=1e-4):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._proba(X)
            err = p - y                                # dBCE/dlogit
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def _proba(self, X):
        z = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict_proba(self, X):
        return self._proba(X)

    def predict(self, X):
        return (self._proba(X) >= 0.5).astype(int)


def make_impressions(n=4000, seed=0):
    # Synthetic ad-impression log. Each ad headline is stitched from token
    # pools with a PLANTED effect on click behavior: "hot" words push clicks
    # up, "cold" words push them down, neutral filler carries no signal.
    # The true click label is sampled from a logistic model of those words,
    # so a from-scratch classifier should recover the latent CTR structure.
    rng = np.random.RandomState(seed)
    hot = ["free", "sale", "now", "win", "deal", "save", "bonus", "today",
           "limited", "exclusive", "gift", "hurry"]          # high-CTR words
    cold = ["terms", "policy", "form", "survey", "invoice", "notice",
            "disclaimer", "subscribe", "renewal", "warranty"]  # low-CTR words
    neutral = ["the", "a", "your", "our", "with", "for", "and", "to",
               "new", "best", "shop", "click", "get", "here", "online"]
    weights = {w: 1.3 for w in hot}
    weights.update({w: -1.4 for w in cold})
    bias = -0.4                                            # base log-odds

    docs, y = [], []
    for _ in range(n):
        n_hot = rng.randint(0, 3)                          # 0-2 hot words
        n_cold = rng.randint(0, 3)                          # 0-2 cold words
        n_fill = rng.randint(4, 8)
        words = (list(rng.choice(hot, n_hot)) +
                 list(rng.choice(cold, n_cold)) +
                 list(rng.choice(neutral, n_fill)))
        rng.shuffle(words)
        logit = bias + sum(weights.get(w, 0.0) for w in words)
        p = 1.0 / (1.0 + np.exp(-logit))
        docs.append(" ".join(words))
        y.append(rng.rand() < p)                            # sample the click
    return docs, np.array(y, dtype=float)


def roc_auc(y_true, scores):
    # AUC = P(score_pos > score_neg); computed from the rank-sum (Mann-Whitney).
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)           # average-tie ignored
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def log_loss(y_true, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


if __name__ == "__main__":
    np.random.seed(0)

    docs, y = make_impressions(n=4000, seed=0)

    # Held-out split: fit CTR model on 70%, evaluate ranking on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Xtr_docs = [docs[i] for i in tr]
    Xte_docs = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = CountVectorizer().fit(Xtr_docs)
    Xtr, Xte = vec.transform(Xtr_docs), vec.transform(Xte_docs)
    model = LogisticRegression(lr=0.5, epochs=300, l2=1e-4).fit(Xtr, ytr)
    p_te = model.predict_proba(Xte)

    auc = roc_auc(yte, p_te)
    ll = log_loss(yte, p_te)
    acc = np.mean((p_te >= 0.5) == yte)

    # Baseline: predict the constant training CTR for every impression.
    base_ctr = ytr.mean()
    base_ll = log_loss(yte, np.full_like(yte, base_ctr))
    base_acc = max(yte.mean(), 1 - yte.mean())            # majority-class acc

    print("Impressions: %d   Train: %d   Test: %d   Vocab: %d"
          % (len(docs), len(tr), len(te), len(vec.vocab)))
    print("Overall click rate: %.3f" % y.mean())
    print("-" * 60)
    print("CTR model   AUC: %.4f   logloss: %.4f   acc: %.4f"
          % (auc, ll, acc))
    print("Baseline    AUC: %.4f   logloss: %.4f   acc: %.4f"
          % (0.5, base_ll, base_acc))
    print("-" * 60)
    # Highest- and lowest-CTR words the model learned (sanity check).
    inv = {j: w for w, j in vec.vocab.items()}
    top = np.argsort(model.w)[::-1][:5]
    bot = np.argsort(model.w)[:5]
    print("Learned high-CTR words:", [inv[j] for j in top])
    print("Learned low-CTR  words:", [inv[j] for j in bot])
    print("-" * 60)
    print("CTR model beats baseline: %s"
          % (auc > 0.5 and ll < base_ll and acc > base_acc))
