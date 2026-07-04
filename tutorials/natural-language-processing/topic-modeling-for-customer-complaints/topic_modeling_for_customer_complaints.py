import numpy as np


def tokenize(text):
    return text.lower().split()


class LDA:
    """Latent Dirichlet Allocation via collapsed Gibbs sampling, from scratch.

    Each document is a mixture of K topics; each topic is a distribution over
    the vocabulary. We never see the topics -- we recover them by repeatedly
    resampling every word's hidden topic from its full conditional:
        p(z=k | .) ~ (n_dk + alpha) * (n_kw + beta) / (n_k + V*beta)
    Counts of "who assigned what" slowly converge to a coherent topic split."""

    def __init__(self, n_topics=4, alpha=0.1, beta=0.05, iters=120, seed=0):
        self.K = n_topics          # number of latent topics to recover
        self.alpha = alpha         # doc-topic Dirichlet prior (sparse mixtures)
        self.beta = beta           # topic-word Dirichlet prior (sparse topics)
        self.iters = iters
        self.seed = seed

    def _build_vocab(self, docs):
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        self.V = len(self.vocab)

    def fit(self, docs):
        rng = np.random.RandomState(self.seed)
        self._build_vocab(docs)
        D, K, V = len(docs), self.K, self.V

        # Encode docs as integer word-id lists.
        self.tokens = [np.array([self.vocab[w] for w in tokenize(d)]) for d in docs]

        # Count tables and initial random topic assignments.
        self.n_dk = np.zeros((D, K))       # topics per document
        self.n_kw = np.zeros((K, V))       # words per topic
        self.n_k = np.zeros(K)             # total words per topic
        self.z = [rng.randint(0, K, len(t)) for t in self.tokens]
        for d in range(D):
            for i, w in enumerate(self.tokens[d]):
                k = self.z[d][i]
                self.n_dk[d, k] += 1
                self.n_kw[k, w] += 1
                self.n_k[k] += 1

        # Collapsed Gibbs sweeps: resample every token's topic in place.
        for _ in range(self.iters):
            for d in range(D):
                zd, wd = self.z[d], self.tokens[d]
                ndk = self.n_dk[d]
                for i in range(len(wd)):
                    w, k = wd[i], zd[i]
                    # Remove current assignment.
                    ndk[k] -= 1; self.n_kw[k, w] -= 1; self.n_k[k] -= 1
                    # Full conditional over topics, then sample.
                    p = (ndk + self.alpha) * (self.n_kw[:, w] + self.beta) \
                        / (self.n_k + V * self.beta)
                    p /= p.sum()
                    k = np.searchsorted(np.cumsum(p), rng.random())
                    # Add new assignment.
                    zd[i] = k
                    ndk[k] += 1; self.n_kw[k, w] += 1; self.n_k[k] += 1
        return self

    def topic_word(self):
        # Smoothed phi: P(word | topic).
        return (self.n_kw + self.beta) / (self.n_k[:, None] + self.V * self.beta)

    def doc_topic(self):
        # Smoothed theta: P(topic | document).
        return (self.n_dk + self.alpha) / \
            (self.n_dk.sum(1, keepdims=True) + self.K * self.alpha)

    def top_words(self, topic, n=6):
        inv = {j: w for w, j in self.vocab.items()}
        return [inv[j] for j in np.argsort(self.topic_word()[topic])[::-1][:n]]

    def transform(self, docs):
        # Dominant topic of each (already-seen) document.
        return np.argmax(self.doc_topic(), axis=1)

    def infer(self, text):
        # Most likely topic for a NEW document: argmax_k sum log P(word|topic).
        ids = [self.vocab[w] for w in tokenize(text) if w in self.vocab]
        if not ids:
            return -1
        logphi = np.log(self.topic_word()[:, ids])
        return int(np.argmax(logphi.sum(axis=1)))


def make_complaints(n=240, seed=0):
    # Synthetic customer complaints. Each complaint is drawn from ONE latent
    # theme (billing / delivery / quality / support) using mostly that theme's
    # words plus shared filler -- so a real topic split exists to be recovered.
    rng = np.random.RandomState(seed)
    themes = {
        0: ["charged", "refund", "invoice", "payment", "overcharged", "fee", "bill", "subscription"],
        1: ["package", "delivery", "late", "shipping", "arrived", "tracking", "courier", "delayed"],
        2: ["broken", "defective", "quality", "damaged", "faulty", "cracked", "poor", "stopped"],
        3: ["agent", "support", "call", "response", "waiting", "rude", "help", "hold"],
    }
    filler = ["the", "my", "was", "and", "is", "very", "this", "with", "order", "please"]
    labels = list(themes.keys())

    docs, y = [], []
    for _ in range(n):
        c = rng.choice(labels)
        length = rng.randint(10, 16)
        n_topic = int(length * 0.65)                 # 65% theme words, 35% filler
        words = list(rng.choice(themes[c], n_topic)) + \
            list(rng.choice(filler, length - n_topic))
        rng.shuffle(words)
        docs.append(" ".join(words))
        y.append(c)
    return docs, np.array(y)


def best_match_accuracy(y_true, z_pred, K):
    # Topics are unlabeled, so score the best permutation mapping topic->theme.
    from itertools import permutations
    best = 0.0
    for perm in permutations(range(K)):
        mapped = np.array([perm[z] for z in z_pred])
        best = max(best, np.mean(mapped == y_true))
    return best


if __name__ == "__main__":
    np.random.seed(0)

    docs, y = make_complaints(n=240, seed=0)
    theme_names = ["Billing", "Delivery", "Quality", "Support"]
    K = 4

    lda = LDA(n_topics=K, iters=120, seed=0).fit(docs)
    z = lda.transform(docs)

    acc = best_match_accuracy(y, z, K)
    # Random baseline: assigning topics by chance recovers ~1/K of the labels.
    random_level = 1.0 / K

    print("Complaints: %d   Vocabulary: %d   Topics: %d"
          % (len(docs), lda.V, K))
    print("-" * 58)
    print("Recovered topics (top words):")
    for k in range(K):
        print("  topic %d: %s" % (k, ", ".join(lda.top_words(k))))
    print("-" * 58)
    print("LDA topic-recovery accuracy: %.4f" % acc)
    print("Random-assignment baseline : %.4f" % random_level)
    print("-" * 58)
    for text in ["overcharged fee refund invoice payment",
                 "package late shipping tracking courier",
                 "broken defective damaged cracked quality",
                 "agent rude support call waiting hold"]:
        print("  '%s...' -> topic %d" % (text[:30], lda.infer(text)))
    print("-" * 58)
    print("LDA beats random topic assignment: %s" % (acc > random_level + 0.15))
