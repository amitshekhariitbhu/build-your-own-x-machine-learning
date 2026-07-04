import numpy as np

class KeywordResearchSystem:
    """From-scratch keyword research: rank candidate keywords by TF-IDF and
    suggest related keywords via PPMI co-occurrence similarity."""

    def __init__(self):
        self.vocab = None      # term -> column index
        self.terms = None      # index -> term
        self.idf = None
        self.tfidf = None      # doc x term tf-idf
        self.freq = None       # raw corpus frequency per term (for baseline)
        self.scores = None     # aggregated keyword score per term
        self.sim = None        # term x term PPMI cosine similarity

    def _tokenize(self, doc):
        return [w for w in doc.lower().split() if w]

    def fit(self, docs):
        # Build vocabulary from the corpus
        toks = [self._tokenize(d) for d in docs]
        vocab = {}
        for t in toks:
            for w in t:
                vocab.setdefault(w, len(vocab))
        V, N = len(vocab), len(docs)
        self.vocab = vocab
        self.terms = np.array(sorted(vocab, key=vocab.get))

        # Term-frequency matrix (doc x term)
        tf = np.zeros((N, V))
        for i, t in enumerate(toks):
            for w in t:
                tf[i, vocab[w]] += 1.0
        self.freq = tf.sum(axis=0)

        # IDF: rare-across-docs terms weigh more; ubiquitous filler -> ~0
        df = (tf > 0).sum(axis=0)
        self.idf = np.log((1.0 + N) / (1.0 + df))
        self.tfidf = tf * self.idf
        # Global keyword score = summed tf-idf across the corpus
        self.scores = self.tfidf.sum(axis=0)

        # Co-occurrence (term x term) over documents, binary presence
        present = (tf > 0).astype(float)
        C = present.T @ present
        np.fill_diagonal(C, 0.0)

        # PPMI weighting suppresses terms that co-occur with everything (fillers)
        total = C.sum() + 1e-12
        p = C / total
        pw = C.sum(axis=1) / total
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(p / (pw[:, None] * pw[None, :] + 1e-12) + 1e-12)
        ppmi = np.maximum(pmi, 0.0)
        ppmi[C == 0] = 0.0

        # Cosine similarity over PPMI vectors -> keyword relatedness
        norm = np.linalg.norm(ppmi, axis=1, keepdims=True) + 1e-12
        U = ppmi / norm
        self.sim = U @ U.T
        return self

    def top_keywords(self, n=10):
        order = np.argsort(-self.scores)[:n]
        return [(self.terms[i], round(float(self.scores[i]), 3)) for i in order]

    def related(self, seed, n=3):
        j = self.vocab[seed]
        s = self.sim[j].copy()
        s[j] = -np.inf
        order = np.argsort(-s)[:n]
        return [str(self.terms[i]) for i in order]


# Example usage
if __name__ == "__main__":
    np.random.seed(0)

    # Planted structure: 4 topics, each with its own signal keywords, plus
    # generic filler words that appear across ALL topics (SEO-style noise).
    topics = {
        "coffee":  ["espresso", "latte", "roast", "bean", "brew", "cappuccino", "grind"],
        "fitness": ["workout", "muscle", "cardio", "protein", "squat", "gym", "stretch"],
        "finance": ["invest", "stock", "dividend", "portfolio", "bond", "saving", "budget"],
        "garden":  ["soil", "seed", "prune", "compost", "bloom", "mulch", "sprout"],
    }
    fillers = ["best", "guide", "top", "tips", "review", "ultimate", "easy", "how"]

    word2topic = {w: t for t, ws in topics.items() for w in ws}
    signal = set(word2topic)
    names = list(topics)

    # Synthetic "search documents": topic keywords buried in filler noise
    docs = []
    for _ in range(40 * len(topics)):
        t = names[np.random.randint(len(names))]
        kw = list(np.random.choice(topics[t], size=5, replace=True))
        # Filler appears in EVERY doc (boilerplate) -> high df -> ~0 IDF
        words = kw + list(fillers)
        np.random.shuffle(words)
        docs.append(" ".join(words))

    krs = KeywordResearchSystem().fit(docs)
    V = len(krs.vocab)

    # ---- Signal 1: keyword ranking recovers planted signal words ----
    K = 20
    top = [w for w, _ in krs.top_keywords(K)]
    prec = np.mean([w in signal for w in top])
    base_order = np.argsort(-krs.freq)[:K]           # baseline: raw frequency
    base_top = [krs.terms[i] for i in base_order]
    base_prec = np.mean([w in signal for w in base_top])

    print("Top 8 keywords (TF-IDF):", krs.top_keywords(8))
    print("Sample related to 'espresso':", krs.related("espresso"))
    print("Sample related to 'stock'   :", krs.related("stock"))
    print()
    print("--- Signal 1: keyword extraction, precision@%d ---" % K)
    print("TF-IDF ranking precision   :", round(float(prec), 3))
    print("Raw-frequency baseline     :", round(float(base_prec), 3))

    # ---- Signal 2: related-keyword suggestion stays on-topic ----
    hits = [np.mean([word2topic.get(r) == word2topic[s] for r in krs.related(s, 3)])
            for s in signal]
    rand = [(len(topics[word2topic[s]]) - 1) / (V - 1) for s in signal]
    sys_prec, rand_prec = float(np.mean(hits)), float(np.mean(rand))
    print()
    print("--- Signal 2: related keywords, same-topic precision@3 ---")
    print("PPMI similarity precision  :", round(sys_prec, 3))
    print("Random baseline            :", round(rand_prec, 3))
    print()
    ok = prec > base_prec and sys_prec > 3 * rand_prec
    print("RESULT:", "PASS - beats both baselines" if ok else "FAIL")
