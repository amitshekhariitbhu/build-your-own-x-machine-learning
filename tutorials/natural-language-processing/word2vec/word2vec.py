import numpy as np


class Word2Vec:
    """Skip-gram Word2Vec with negative sampling, trained by manual SGD.

    Two embedding tables (center W_in, context W_out) are learned so that a
    word's vector predicts the words that co-occur with it. Similar contexts
    -> similar center vectors, so topically-related words end up nearby."""

    def __init__(self, dim=24, window=2, neg=5, epochs=6, lr=0.05, seed=0):
        self.dim = dim          # embedding dimension
        self.window = window    # context radius around each center word
        self.neg = neg          # negative samples per positive pair
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    def _pairs(self, corpus):
        # Flatten sentences into (center, context) skip-gram pairs.
        c, o = [], []
        for sent in corpus:
            for i, center in enumerate(sent):
                lo, hi = max(0, i - self.window), min(len(sent), i + self.window + 1)
                for j in range(lo, hi):
                    if j != i:
                        c.append(center)
                        o.append(sent[j])
        return np.array(c), np.array(o)

    def fit(self, corpus, vocab_size):
        rng = np.random.RandomState(self.seed)
        self.V = vocab_size

        # Noise distribution for negatives: unigram counts ^ 0.75 (Mikolov).
        counts = np.ones(vocab_size)
        for sent in corpus:
            for w in sent:
                counts[w] += 1
        noise = counts ** 0.75
        self.noise = noise / noise.sum()

        # Small random init for input table; zeros for output table.
        self.W_in = (rng.rand(vocab_size, self.dim) - 0.5) / self.dim
        self.W_out = np.zeros((vocab_size, self.dim))

        centers, contexts = self._pairs(corpus)
        n = len(centers)

        for _ in range(self.epochs):
            perm = rng.permutation(n)
            centers, contexts = centers[perm], contexts[perm]

            # One negative-sample draw per pair (vectorized over the whole epoch).
            negs = rng.choice(vocab_size, size=(n, self.neg), p=self.noise)

            for c, o, neg in zip(centers, contexts, negs):
                # Targets = 1 positive context word + neg negatives.
                targets = np.concatenate(([o], neg))
                labels = np.zeros(self.neg + 1)
                labels[0] = 1.0

                h = self.W_in[c]                     # center vector (dim,)
                v = self.W_out[targets]              # target vectors (k+1, dim)
                score = 1.0 / (1.0 + np.exp(-(v @ h)))   # sigmoid predictions
                g = (score - labels)                 # gradient of loss wrt score

                # SGD updates for the touched rows only.
                grad_h = g @ v                       # (dim,)
                self.W_out[targets] -= self.lr * np.outer(g, h)
                self.W_in[c] -= self.lr * grad_h
        return self

    def vectors(self):
        # Center embeddings, L2-normalized (use for similarity / clustering).
        norm = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        return self.W_in / np.maximum(norm, 1e-9)

    def most_similar(self, w, n=3):
        E = self.vectors()
        sims = E @ E[w]
        sims[w] = -np.inf
        return np.argsort(sims)[::-1][:n]


if __name__ == "__main__":
    np.random.seed(0)

    # Planted structure: 4 topics, 8 words each. Sentences are drawn from a
    # single topic, so same-topic words co-occur and should learn close vectors.
    n_topics, per_topic = 4, 8
    V = n_topics * per_topic
    topic_of = np.repeat(np.arange(n_topics), per_topic)   # true label per word
    topic_words = [np.arange(t * per_topic, (t + 1) * per_topic) for t in range(n_topics)]

    corpus, sent_len, n_sent = [], 8, 600
    for _ in range(n_sent):
        t = np.random.randint(n_topics)
        # Skewed within-topic frequencies make some words more common than others.
        p = np.random.rand(per_topic) + 0.2
        p /= p.sum()
        corpus.append(list(np.random.choice(topic_words[t], size=sent_len, p=p)))

    model = Word2Vec(dim=24, window=2, neg=5, epochs=6, lr=0.05).fit(corpus, V)
    E = model.vectors()

    # Correctness: is each word's nearest neighbor in the SAME planted topic?
    hits = 0
    for w in range(V):
        nn = model.most_similar(w, n=1)[0]
        hits += (topic_of[nn] == topic_of[w])
    nn_acc = hits / V

    # Same- vs cross-topic average cosine similarity (should separate clearly).
    S = E @ E.T
    same = np.array([S[i, j] for i in range(V) for j in range(V)
                     if i != j and topic_of[i] == topic_of[j]]).mean()
    cross = np.array([S[i, j] for i in range(V) for j in range(V)
                      if topic_of[i] != topic_of[j]]).mean()

    # Random baseline for the NN test = P(random other word shares the topic).
    baseline = (per_topic - 1) / (V - 1)

    print("Vocab: %d words, %d topics x %d words" % (V, n_topics, per_topic))
    print("Skip-gram pairs trained: %d" % len(model._pairs(corpus)[0]))
    print("Nearest-neighbor same-topic accuracy: %.3f" % nn_acc)
    print("Random baseline (chance):             %.3f" % baseline)
    print("Avg cosine  same-topic: %.3f" % same)
    print("Avg cosine cross-topic: %.3f" % cross)
    print("Same > cross similarity: %s" % (same > cross))
    print("Neighbors of word 0 (topic 0): %s" % model.most_similar(0, n=3).tolist())
    print("Beats baseline: %s" % (nn_acc > baseline))
