import numpy as np


class NGramLanguageModel:
    """Next-word predictor: interpolated trigram language model, from scratch.

    Learns unigram/bigram/trigram counts, then scores the next word with a
    linear interpolation  p(w|a,b) = l3*p_tri + l2*p_bi + l1*p_uni. Longer
    contexts (trigram) capture more structure; shorter ones (uni/bigram) back
    them up when a context was never seen. Unigram is add-k smoothed so no
    word ever gets zero probability."""

    def __init__(self, vocab_size, lambdas=(0.1, 0.3, 0.6), k=0.1):
        self.V = vocab_size
        self.l1, self.l2, self.l3 = lambdas   # uni, bi, tri weights
        self.k = k                            # add-k smoothing for unigram

    def fit(self, sequences):
        V = self.V
        self.uni = np.zeros(V)
        self.bi = np.zeros((V, V))
        self.tri = np.zeros((V, V, V))
        for s in sequences:
            for i, w in enumerate(s):
                self.uni[w] += 1
                if i >= 1:
                    self.bi[s[i - 1], w] += 1
                if i >= 2:
                    self.tri[s[i - 2], s[i - 1], w] += 1
        # Smoothed unigram distribution (always valid, never zero).
        self.p_uni = (self.uni + self.k) / (self.uni.sum() + self.k * V)
        return self

    def dist(self, a, b):
        """Full next-word distribution given the last two words (a, b)."""
        comps, weights = [self.p_uni], [self.l1]
        bsum = self.bi[b].sum()
        if bsum > 0:
            comps.append(self.bi[b] / bsum)
            weights.append(self.l2)
        tsum = self.tri[a, b].sum()
        if tsum > 0:
            comps.append(self.tri[a, b] / tsum)
            weights.append(self.l3)
        w = np.array(weights) / sum(weights)   # renormalize used weights
        return np.tensordot(w, np.array(comps), axes=1)

    def predict_next(self, a, b):
        return int(np.argmax(self.dist(a, b)))

    def perplexity(self, sequences):
        # exp(mean negative log-prob of each realized next word).
        logp, n = 0.0, 0
        for s in sequences:
            for i in range(2, len(s)):
                logp += np.log(self.dist(s[i - 2], s[i - 1])[s[i]] + 1e-12)
                n += 1
        return float(np.exp(-logp / n))

    def accuracy(self, sequences):
        # Top-1 next-word accuracy at every trigram position.
        hits, n = 0, 0
        for s in sequences:
            for i in range(2, len(s)):
                hits += self.predict_next(s[i - 2], s[i - 1]) == s[i]
                n += 1
        return hits / n


def make_corpus(V=30, fanout=3, n_sent=400, length=12, seed=0):
    """Plant a peaked word-level Markov chain and sample sentences from it.

    Each word transitions to a few successors with a skewed (peaked) law, so
    the *next word is genuinely predictable from context* -- the latent
    structure an n-gram model must recover. No single global word dominates,
    so a unigram baseline cannot cheat."""
    rng = np.random.RandomState(seed)
    T = np.zeros((V, V))
    for i in range(V):
        succ = rng.choice(V, size=fanout, replace=False)
        probs = np.sort(rng.dirichlet(np.ones(fanout) * 0.3))[::-1]  # peaked
        T[i, succ] = probs
    corpus = []
    for _ in range(n_sent):
        w = rng.randint(V)
        s = [w]
        for _ in range(length - 1):
            w = rng.choice(V, p=T[w])
            s.append(w)
        corpus.append(s)
    return corpus, T


if __name__ == "__main__":
    np.random.seed(0)

    V = 30
    corpus, T = make_corpus(V=V, fanout=3, n_sent=400, length=12, seed=0)

    # Held-out split.
    split = int(0.8 * len(corpus))
    train, test = corpus[:split], corpus[split:]

    model = NGramLanguageModel(vocab_size=V, lambdas=(0.1, 0.3, 0.6), k=0.1).fit(train)

    acc = model.accuracy(test)
    ppl = model.perplexity(test)

    # Baseline 1: always predict the globally most frequent word (unigram argmax).
    top_word = int(np.argmax(model.uni))
    base_hits = base_n = 0
    for s in test:
        for i in range(2, len(s)):
            base_hits += (s[i] == top_word)
            base_n += 1
    base_acc = base_hits / base_n
    rand_acc = 1.0 / V

    # Baseline perplexity: unigram-only model on the same test set.
    uni_logp = uni_n = 0.0
    for s in test:
        for i in range(2, len(s)):
            uni_logp += np.log(model.p_uni[s[i]] + 1e-12)
            uni_n += 1
    uni_ppl = float(np.exp(-uni_logp / uni_n))

    print("Vocab: %d words | train sents: %d | test sents: %d" % (V, len(train), len(test)))
    print("Next-word top-1 accuracy (trigram): %.3f" % acc)
    print("Baseline accuracy (most-freq word): %.3f" % base_acc)
    print("Random baseline (1/V):              %.3f" % rand_acc)
    print("Perplexity (trigram):  %.2f" % ppl)
    print("Perplexity (unigram):  %.2f" % uni_ppl)
    ctx = test[0][:2]
    print("Example: after words %s -> predict %d" % (ctx, model.predict_next(ctx[0], ctx[1])))
    print("Beats freq baseline: %s" % (acc > base_acc))
    print("Beats random & lower perplexity: %s" % (acc > rand_acc and ppl < uni_ppl))
