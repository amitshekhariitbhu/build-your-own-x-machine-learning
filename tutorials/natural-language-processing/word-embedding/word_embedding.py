import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class Word2Vec:
    # Skip-gram with negative sampling, trained from scratch with numpy.
    def __init__(self, vocab_size, dim=24, window=2, n_neg=5,
                 lr=0.05, epochs=40, batch=512):
        self.V, self.dim, self.window = vocab_size, dim, window
        self.n_neg, self.lr, self.epochs, self.batch = n_neg, lr, epochs, batch
        # Two embedding tables: input (center) and output (context) vectors.
        self.W_in = (np.random.rand(vocab_size, dim) - 0.5) / dim
        self.W_out = (np.random.rand(vocab_size, dim) - 0.5) / dim

    def _pairs(self, sentences):
        # Build all (center, context) index pairs inside the window.
        centers, contexts = [], []
        for s in sentences:
            for i, c in enumerate(s):
                lo, hi = max(0, i - self.window), min(len(s), i + self.window + 1)
                for j in range(lo, hi):
                    if j != i:
                        centers.append(c)
                        contexts.append(s[j])
        return np.array(centers), np.array(contexts)

    def fit(self, sentences):
        c_all, o_all = self._pairs(sentences)
        # Negative-sampling distribution: unigram frequency ^ 0.75.
        freq = np.bincount(np.concatenate([s for s in sentences]), minlength=self.V)
        p = (freq.astype(np.float64) ** 0.75)
        p /= p.sum()
        n = len(c_all)
        for _ in range(self.epochs):
            order = np.random.permutation(n)
            for start in range(0, n, self.batch):
                idx = order[start:start + self.batch]
                c, o = c_all[idx], o_all[idx]                 # (B,), (B,)
                neg = np.random.choice(self.V, size=(len(c), self.n_neg), p=p)
                vc, vo = self.W_in[c], self.W_out[o]          # (B,d)
                vn = self.W_out[neg]                          # (B,K,d)
                # Positive: label 1; negatives: label 0.
                gpos = sigmoid(np.sum(vc * vo, axis=1)) - 1.0        # (B,)
                gneg = sigmoid(np.einsum('bd,bkd->bk', vc, vn))      # (B,K)
                grad_vo = gpos[:, None] * vc                          # (B,d)
                grad_vn = gneg[:, :, None] * vc[:, None, :]           # (B,K,d)
                grad_vc = gpos[:, None] * vo + np.einsum('bk,bkd->bd', gneg, vn)
                # Scatter-add gradients (indices repeat, so use np.add.at).
                self.W_out[o] -= self.lr * grad_vo
                np.add.at(self.W_out, neg, -self.lr * grad_vn)
                np.add.at(self.W_in, c, -self.lr * grad_vc)
        return self

    def vectors(self):
        return self.W_in


def make_corpus(n_topics=5, words_per_topic=8, n_sentences=600):
    # Planted structure: words split into topics; each sentence samples
    # words from ONE topic, so same-topic words co-occur inside windows.
    V = n_topics * words_per_topic
    topic_of = np.repeat(np.arange(n_topics), words_per_topic)
    sentences = []
    for _ in range(n_sentences):
        t = np.random.randint(n_topics)
        pool = np.where(topic_of == t)[0]
        length = np.random.randint(8, 13)
        sentences.append(np.random.choice(pool, size=length))
    return sentences, topic_of, V


def neighbor_topic_accuracy(vecs, topic_of):
    # For each word, is its nearest neighbor (cosine) in the same topic?
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argmax(sim, axis=1)
    return np.mean(topic_of[nn] == topic_of)


if __name__ == "__main__":
    np.random.seed(0)
    sentences, topic_of, V = make_corpus()

    model = Word2Vec(V).fit(sentences)
    vecs = model.vectors()

    acc = neighbor_topic_accuracy(vecs, topic_of)
    # Random baseline: chance that a random other word shares the topic.
    words_per_topic = np.sum(topic_of == 0)
    baseline = (words_per_topic - 1) / (V - 1)

    # Semantic geometry check: intra-topic vs inter-topic similarity.
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    same = topic_of[:, None] == topic_of[None, :]
    off = ~np.eye(V, dtype=bool)
    intra = sim[same & off].mean()
    inter = sim[~same].mean()

    print("Vocabulary size:            ", V)
    print("Embedding dim:              ", model.dim)
    print("Topics:                     ", len(np.unique(topic_of)))
    print("-" * 44)
    print("Mean intra-topic cosine:     %.3f" % intra)
    print("Mean inter-topic cosine:     %.3f" % inter)
    print("-" * 44)
    print("Nearest-neighbor topic acc:  %.3f" % acc)
    print("Random baseline:             %.3f" % baseline)
    print("Beats baseline:             ", bool(acc > baseline + 0.3))
