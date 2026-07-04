import numpy as np

# ---- Word lists the "grader" knows about (its internal dictionary). ----
FILLER = ["the", "a", "of", "in", "to", "and", "is", "that", "it", "we"]
SIMPLE = ["good", "nice", "big", "thing", "make", "very", "lot", "many", "get", "way"]
SOPHISTICATED = ["consequently", "furthermore", "significant", "perspective",
                 "demonstrate", "substantial", "analysis", "framework",
                 "compelling", "nuanced", "articulate", "underscore",
                 "empirical", "profound", "meticulous", "coherent"]
CONNECTIVES = ["however", "therefore", "moreover", "nevertheless",
               "consequently", "furthermore", "whereas", "thus"]
KNOWN = set(FILLER + SIMPLE + SOPHISTICATED + CONNECTIVES)
SOPH_SET, CONN_SET = set(SOPHISTICATED), set(CONNECTIVES)


def _corrupt(word, rng):
    # Turn a word into an out-of-dictionary "misspelling" by swapping two chars.
    if len(word) < 4:
        return word + "x"
    i = rng.randint(0, len(word) - 2)
    c = list(word)
    c[i], c[i + 1] = c[i + 1], c[i]
    return "".join(c)


def make_essays(n=400, seed=0):
    # Synthetic essays: a latent quality q in [1,6] controls length, vocabulary
    # sophistication, connective use and spelling errors -> planted, recoverable.
    rng = np.random.RandomState(seed)
    texts, grades = [], []
    for _ in range(n):
        q = rng.uniform(1.0, 6.0)                       # latent quality
        n_words = int(20 + 14 * q + rng.normal(0, 5))
        n_words = max(15, n_words)
        p_soph = 0.05 + 0.10 * q                        # better essays: richer words
        p_conn = 0.01 + 0.025 * q
        typo_rate = max(0.0, 0.14 - 0.020 * q)          # better essays: fewer typos
        sent_len = 7 + int(q)                           # longer sentences when better

        words = []
        for k in range(n_words):
            r = rng.rand()
            if r < p_conn:
                w = rng.choice(CONNECTIVES)
            elif r < p_conn + p_soph:
                w = rng.choice(SOPHISTICATED)
            elif r < p_conn + p_soph + 0.35:
                w = rng.choice(SIMPLE)
            else:
                w = rng.choice(FILLER)
            if rng.rand() < typo_rate:
                w = _corrupt(w, rng)
            words.append(w)
            if (k + 1) % sent_len == 0:
                words.append(".")
        texts.append(" ".join(words))
        grades.append(int(np.clip(round(q), 1, 6)))
    return texts, np.array(grades, dtype=float)


def features(text):
    # Hand-crafted essay features, all derived only from the raw text.
    toks = text.split()
    words = [t for t in toks if t != "."]
    n = max(1, len(words))
    n_sent = max(1, sum(t == "." for t in toks))
    unique = len(set(words))
    soph = sum(w in SOPH_SET for w in words)
    conn = sum(w in CONN_SET for w in words)
    misspell = sum(w not in KNOWN for w in words)      # OOV token = spelling error
    avg_len = np.mean([len(w) for w in words])
    return np.array([n,                                 # essay length
                     unique / n,                        # type-token ratio
                     avg_len,                           # avg word length
                     soph / n,                          # sophisticated-word rate
                     conn / n,                          # connective rate
                     misspell / n,                      # misspelling rate
                     n / n_sent])                       # avg sentence length


class RidgeRegressor:
    """Least-squares grader with L2 regularization via the normal equations."""

    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9   # standardize features
        Xs = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        R = self.lam * np.eye(Xs.shape[1])
        R[0, 0] = 0.0                                   # do not penalize the bias
        self.w = np.linalg.solve(Xs.T @ Xs + R, Xs.T @ y)
        return self

    def predict(self, X):
        Xs = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        return Xs @ self.w


def quadratic_weighted_kappa(a, b, lo=1, hi=6):
    # Standard essay-grading agreement metric; 0 = chance, 1 = perfect.
    a, b = a.astype(int), b.astype(int)
    K = hi - lo + 1
    O = np.zeros((K, K))
    for x, y in zip(a - lo, b - lo):
        O[x, y] += 1
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    hist_a = O.sum(1)
    hist_b = O.sum(0)
    E = np.outer(hist_a, hist_b) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()


if __name__ == "__main__":
    np.random.seed(0)

    texts, grades = make_essays(n=400, seed=0)
    X = np.array([features(t) for t in texts])

    idx = np.random.permutation(len(texts))             # held-out split
    split = int(0.7 * len(texts))
    tr, te = idx[:split], idx[split:]

    model = RidgeRegressor(lam=1.0).fit(X[tr], grades[tr])
    pred = model.predict(X[te])
    pred_grade = np.clip(np.round(pred), 1, 6)

    base = np.full(len(te), grades[tr].mean())          # predict-the-mean baseline
    rmse = np.sqrt(np.mean((pred - grades[te]) ** 2))
    base_rmse = np.sqrt(np.mean((base - grades[te]) ** 2))
    mae = np.mean(np.abs(pred - grades[te]))

    majority = np.bincount(grades[tr].astype(int)).argmax()
    base_grade = np.full(len(te), majority, dtype=float)
    acc = np.mean(pred_grade == grades[te])
    base_acc = np.mean(base_grade == grades[te])
    qwk = quadratic_weighted_kappa(pred_grade, grades[te])

    print("Essays: %d   Train: %d   Test: %d   Grades: 1-6" % (len(texts), len(tr), len(te)))
    print("Features: length, type-token, word-len, sophistication, connectives, misspell, sent-len")
    print("-" * 60)
    print("Ridge grader   RMSE: %.3f   MAE: %.3f   exact-acc: %.3f" % (rmse, mae, acc))
    print("Predict-mean   RMSE: %.3f              exact-acc: %.3f" % (base_rmse, base_acc))
    print("Quadratic weighted kappa (vs human): %.3f   (0 = chance)" % qwk)
    print("-" * 60)
    for t in texts[:1] + [texts[np.argmax(grades)]]:
        print("  grade %.1f  <- \"%s ...\"" % (model.predict([features(t)])[0], t[:52]))
    print("-" * 60)
    print("Grader beats baseline: %s" % (rmse < base_rmse and acc > base_acc and qwk > 0.5))
