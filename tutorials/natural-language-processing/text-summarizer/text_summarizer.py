import numpy as np


def tokenize(sentence):
    return sentence.split()


class TextRankSummarizer:
    """Extractive summarizer: TF-IDF sentence graph + PageRank centrality."""

    def __init__(self, damping=0.85, n_iter=80, tol=1e-8):
        self.damping = damping      # PageRank teleport probability = 1 - damping
        self.n_iter = n_iter
        self.tol = tol

    def _tfidf(self, sentences):
        # Build a vocabulary over the document's sentences.
        vocab = {}
        for s in sentences:
            for w in tokenize(s):
                vocab.setdefault(w, len(vocab))
        N, V = len(sentences), len(vocab)

        tf = np.zeros((N, V))
        for i, s in enumerate(sentences):
            for w in tokenize(s):
                tf[i, vocab[w]] += 1.0

        df = (tf > 0).sum(0)                        # sentences containing each word
        idf = np.log((1.0 + N) / (1.0 + df)) + 1.0  # rare words weigh more
        X = tf * idf[None, :]

        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return X / norm                             # L2-normalized -> dot = cosine

    def _pagerank(self, S):
        # Power iteration on the column-stochastic transition matrix.
        N = S.shape[0]
        col = S.sum(0)
        col[col == 0] = 1.0
        M = S / col[None, :]
        r = np.ones(N) / N
        for _ in range(self.n_iter):
            r_new = (1.0 - self.damping) / N + self.damping * (M @ r)
            if np.abs(r_new - r).sum() < self.tol:
                return r_new
            r = r_new
        return r

    def fit(self, sentences):
        self.sentences = sentences
        X = self._tfidf(sentences)
        S = X @ X.T                                 # sentence-sentence cosine graph
        np.fill_diagonal(S, 0.0)                    # no self-similarity edges
        S[S < 0] = 0.0
        self.scores = self._pagerank(S)             # centrality = importance
        return self

    def summarize(self, k):
        # Top-k most central sentences, returned in original reading order.
        idx = np.argsort(self.scores)[::-1][:k]
        return np.sort(idx)


def make_documents(n_docs=40, n_topics=5, n_key=4, n_filler=8, seed=0):
    # Each doc has one theme: KEY sentences pack that topic's rare core words
    # (so they cluster tightly); FILLER sentences are generic + off-topic noise.
    rng = np.random.RandomState(seed)
    core = {t: ["t%d_w%d" % (t, j) for j in range(8)] for t in range(n_topics)}
    generic = ["g%02d" % j for j in range(40)]      # background words, low IDF

    docs, key_sets = [], []
    for _ in range(n_docs):
        t = rng.randint(n_topics)
        sents = []
        for _ in range(n_key):                      # on-theme: 6 core + 2 generic
            w = list(rng.choice(core[t], 6)) + list(rng.choice(generic, 2))
            rng.shuffle(w)
            sents.append(" ".join(w))
        for _ in range(n_filler):                   # off-theme: 6 generic + 2 other
            o = rng.choice([x for x in range(n_topics) if x != t])
            w = list(rng.choice(generic, 6)) + list(rng.choice(core[o], 2))
            rng.shuffle(w)
            sents.append(" ".join(w))
        order = rng.permutation(len(sents))         # shuffle reading order
        docs.append([sents[i] for i in order])
        key_sets.append(set(np.where(order < n_key)[0].tolist()))
    return docs, key_sets


def evaluate(summarizer, docs, key_sets, k):
    # F1 between the k extracted sentences and the planted key set, averaged.
    f1s, base = [], []
    for sents, key in zip(docs, key_sets):
        pred = set(summarizer.fit(sents).summarize(k).tolist())
        tp = len(pred & key)
        prec, rec = tp / k, tp / len(key)
        f1s.append(0.0 if tp == 0 else 2 * prec * rec / (prec + rec))
        base.append(len(key) / len(sents))          # random Precision=Recall=F1
    return float(np.mean(f1s)), float(np.mean(base))


if __name__ == "__main__":
    np.random.seed(0)

    N_KEY = 4
    docs, key_sets = make_documents(n_docs=40, n_topics=5, n_key=N_KEY, seed=0)
    n_sent = len(docs[0])

    summ = TextRankSummarizer()
    f1, rand_f1 = evaluate(summ, docs, key_sets, k=N_KEY)

    print("Docs: %d   Sentences/doc: %d   Key sentences/doc: %d"
          % (len(docs), n_sent, N_KEY))
    print("-" * 56)
    print("TextRank summary F1 :  %.4f  (%.1fx random)" % (f1, f1 / rand_f1))
    print("Random baseline F1  :  %.4f" % rand_f1)
    print("-" * 56)

    # Show one worked example: which sentences the summarizer picked.
    sents, key = docs[0], sorted(key_sets[0])
    picked = summ.fit(sents).summarize(N_KEY).tolist()
    print("Doc 0 planted key sentences  : %s" % key)
    print("Doc 0 TextRank chose         : %s" % picked)
    print("Doc 0 example key sentence   : \"%s\"" % sents[key[0]])
    print("-" * 56)
    print("Summarizer beats random: %s" % (f1 > rand_f1))
