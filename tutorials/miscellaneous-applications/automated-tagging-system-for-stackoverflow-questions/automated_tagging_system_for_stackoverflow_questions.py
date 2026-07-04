import numpy as np

class AutomatedTagger:
    """From-scratch multi-label tagger for StackOverflow questions.
    TF-IDF text features + one-vs-rest logistic regression (one binary
    classifier per tag), trained by vectorized gradient descent."""

    def __init__(self, lr=0.5, epochs=300, l2=1e-3, thresh=0.5):
        self.lr, self.epochs, self.l2, self.thresh = lr, epochs, l2, thresh
        self.vocab = None    # term -> column index
        self.idf = None
        self.W = None        # (V+1) x K weights (last row = bias)

    def _tokenize(self, doc):
        return [w for w in doc.lower().split() if w]

    def _tfidf(self, docs, fit):
        toks = [self._tokenize(d) for d in docs]
        if fit:
            vocab = {}
            for t in toks:
                for w in t:
                    vocab.setdefault(w, len(vocab))
            self.vocab = vocab
        V = len(self.vocab)
        tf = np.zeros((len(docs), V))
        for i, t in enumerate(toks):
            for w in t:
                j = self.vocab.get(w)
                if j is not None:
                    tf[i, j] += 1.0
        if fit:
            df = (tf > 0).sum(axis=0)
            self.idf = np.log((1.0 + len(docs)) / (1.0 + df)) + 1.0
        X = tf * self.idf
        norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norm

    def fit(self, docs, Y):
        X = self._tfidf(docs, fit=True)
        X = np.hstack([X, np.ones((X.shape[0], 1))])     # bias column
        N, D = X.shape
        self.W = np.zeros((D, Y.shape[1]))
        for _ in range(self.epochs):
            P = 1.0 / (1.0 + np.exp(-(X @ self.W)))
            grad = X.T @ (P - Y) / N + self.l2 * self.W
            self.W -= self.lr * grad
        return self

    def predict_proba(self, docs):
        X = self._tfidf(docs, fit=False)
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        return 1.0 / (1.0 + np.exp(-(X @ self.W)))

    def predict(self, docs):
        return (self.predict_proba(docs) >= self.thresh).astype(int)


def micro_f1(Y_true, Y_pred):
    tp = np.sum((Y_pred == 1) & (Y_true == 1))
    fp = np.sum((Y_pred == 1) & (Y_true == 0))
    fn = np.sum((Y_pred == 0) & (Y_true == 1))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    return 2 * prec * rec / (prec + rec + 1e-12)


def make_dataset(n, tags, tag_words, filler, rng):
    """Each question gets 1-2 tags; words drawn from those tags' vocab + noise."""
    docs, Y = [], np.zeros((n, len(tags)), dtype=int)
    for i in range(n):
        k = rng.integers(1, 3)                       # 1 or 2 tags per question
        chosen = rng.choice(len(tags), size=k, replace=False)
        words = []
        for c in chosen:
            Y[i, c] = 1
            words += list(rng.choice(tag_words[tags[c]], size=rng.integers(4, 8)))
        words += list(rng.choice(filler, size=rng.integers(3, 7)))  # noise
        rng.shuffle(words)
        docs.append(" ".join(words))
    return docs, Y


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # Planted structure: 6 tags, each with its own signal vocabulary.
    tags = ["python", "javascript", "sql", "html-css", "machine-learning", "android"]
    tag_words = {
        "python": "python pandas numpy list dict lambda pip virtualenv django flask".split(),
        "javascript": "javascript node npm react async promise dom event const arrow".split(),
        "sql": "sql select join query table index postgres mysql where group".split(),
        "html-css": "html css flexbox div margin padding selector responsive grid font".split(),
        "machine-learning": "model train tensor gradient neural loss accuracy dataset epoch layer".split(),
        "android": "android kotlin activity intent fragment gradle recyclerview view layout manifest".split(),
    }
    filler = "how do i fix this error code please help thanks question issue when using".split()

    docs_tr, Y_tr = make_dataset(600, tags, tag_words, filler, rng)
    docs_te, Y_te = make_dataset(200, tags, tag_words, filler, rng)

    model = AutomatedTagger().fit(docs_tr, Y_tr)
    Y_pred = model.predict(docs_te)

    # Baseline: always predict the globally most common tag (majority label).
    common = np.argmax(Y_tr.sum(axis=0))
    Y_base = np.zeros_like(Y_te)
    Y_base[:, common] = 1

    f1_model = micro_f1(Y_te, Y_pred)
    f1_base = micro_f1(Y_te, Y_base)
    subset_acc = np.mean(np.all(Y_pred == Y_te, axis=1))

    print(f"Tags: {tags}")
    print(f"Test questions: {len(docs_te)}  (multi-label)")
    print(f"Baseline micro-F1 (majority tag): {f1_base:.3f}")
    print(f"Tagger   micro-F1               : {f1_model:.3f}")
    print(f"Exact-set accuracy (all tags right): {subset_acc:.3f}")
    print("PASS" if f1_model > f1_base + 0.3 else "FAIL")

    # Show tags predicted for a few held-out questions.
    for d, yp, yt in list(zip(docs_te, Y_pred, Y_te))[:3]:
        pt = [tags[j] for j in np.where(yp == 1)[0]]
        gt = [tags[j] for j in np.where(yt == 1)[0]]
        print(f"\nQ: {d[:70]}...\n  predicted={pt}  true={gt}")
