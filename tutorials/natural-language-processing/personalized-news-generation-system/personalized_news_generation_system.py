import numpy as np

# --- Planted topic vocabularies (the latent structure to recover) ----------
# Each topic owns distinctive content words; templates share function words.
# A per-topic bigram model learns to REGENERATE these; a classifier keys off
# the content words to tell which topic a generated headline belongs to.
TOPIC_WORDS = {
    0: {"actor":  ["team", "player", "coach", "striker", "keeper"],       # sports
        "action": ["defeated", "beat", "outscored", "dominated"],
        "target": ["rivals", "champions", "opponents", "underdogs"],
        "extra":  ["match", "goal", "season", "tournament", "stadium"]},
    1: {"actor":  ["startup", "company", "engineer", "chipmaker", "team"],  # tech
        "action": ["launched", "shipped", "unveiled", "patented"],
        "target": ["gadget", "platform", "app", "processor"],
        "extra":  ["software", "network", "cloud", "device", "data"]},
    2: {"actor":  ["market", "investor", "bank", "trader", "fund"],        # finance
        "action": ["rallied", "surged", "tumbled", "hedged"],
        "target": ["stocks", "bonds", "shares", "assets"],
        "extra":  ["profit", "interest", "dividend", "earnings", "rate"]},
    3: {"actor":  ["senator", "minister", "party", "governor", "council"],  # politics
        "action": ["proposed", "vetoed", "debated", "passed"],
        "target": ["bill", "policy", "reform", "budget"],
        "extra":  ["election", "vote", "law", "campaign", "summit"]},
}
TEMPLATES = [
    ["the", "{actor}", "{action}", "the", "{target}", "in", "the", "{extra}"],
    ["{actor}", "{action}", "{target}", "during", "the", "{extra}"],
    ["reports", "say", "the", "{actor}", "{action}", "the", "{target}"],
    ["a", "{actor}", "{action}", "the", "{target}", "and", "the", "{extra}"],
]


def make_corpus(n_per_topic=160, seed=0):
    """Sample templated headlines per topic -> planted, recoverable topics."""
    rng = np.random.RandomState(seed)
    topics, sents = [], []
    for t, slots in TOPIC_WORDS.items():
        for _ in range(n_per_topic):
            tpl = TEMPLATES[rng.randint(len(TEMPLATES))]
            s = [slots[w[1:-1]][rng.randint(len(slots[w[1:-1]]))]
                 if w.startswith("{") else w for w in tpl]
            topics.append(t)
            sents.append(s)
    order = rng.permutation(len(sents))
    return np.array(topics)[order], [sents[i] for i in order]


class NaiveBayes:
    """Multinomial bag-of-words Naive Bayes (add-1 smoothed), from scratch.
    Used as an INDEPENDENT judge of which topic a generated headline reads as."""

    def fit(self, X, y, V, C):
        self.logprior = np.log(np.bincount(y, minlength=C) / len(y))
        counts = np.ones((C, V))                       # Laplace add-1
        for ids, c in zip(X, y):
            np.add.at(counts[c], ids, 1)
        self.loglik = np.log(counts / counts.sum(1, keepdims=True))
        return self

    def predict(self, X):
        out = np.empty(len(X), dtype=int)
        for i, ids in enumerate(X):
            out[i] = np.argmax(self.logprior + self.loglik[:, ids].sum(1))
        return out


class PersonalizedNewsGenerator:
    """Per-topic bigram language models + a user-interest topic sampler.

    fit()  learns one START->..->END bigram transition matrix per topic.
    A user is a preference distribution over topics; to write their feed we
    sample a topic from that distribution, then roll out its bigram chain."""

    def __init__(self, n_topics, V):
        self.C, self.V = n_topics, V
        self.START, self.END = V, V + 1                # two special ids

    def fit(self, X, y):
        Z = self.V + 2
        self.trans = np.zeros((self.C, Z, Z))
        for ids, c in zip(X, y):
            seq = [self.START, *ids, self.END]
            for a, b in zip(seq[:-1], seq[1:]):
                self.trans[c, a, b] += 1
        return self

    def generate(self, topic, rng, max_len=12):
        cur, out = self.START, []
        for _ in range(max_len):
            row = self.trans[topic, cur]
            s = row.sum()
            if s == 0:
                break
            nxt = rng.choice(row.size, p=row / s)
            if nxt == self.END:
                break
            out.append(nxt)
            cur = nxt
        return out

    def feed(self, pref, n, rng):
        """Generate n headlines for a user with topic-preference vector pref."""
        arts = []
        for _ in range(n):
            t = rng.choice(self.C, p=pref)
            arts.append(self.generate(t, rng))
        return arts


if __name__ == "__main__":
    np.random.seed(0)
    C = len(TOPIC_WORDS)

    topics, sents = make_corpus(n_per_topic=160, seed=0)
    vocab = {w: i for i, w in enumerate(sorted({w for s in sents for w in s}))}
    inv = {i: w for w, i in vocab.items()}
    X = [np.array([vocab[w] for w in s]) for s in sents]
    V = len(vocab)

    # Held-out split for the topic judge.
    split = int(0.8 * len(X))
    clf = NaiveBayes().fit(X[:split], topics[:split], V, C)
    acc = (clf.predict(X[split:]) == topics[split:]).mean()
    majority = np.bincount(topics).max() / len(topics)   # best constant guess

    gen = PersonalizedNewsGenerator(C, V).fit(X, topics)

    # Users: each favors one topic (0.85) with a little spread (0.05 each).
    rng = np.random.RandomState(1)
    n_users, per_user = 12, 40
    favs = np.arange(n_users) % C
    hit_pers = hit_base = 0
    example = None
    for u, fav in enumerate(favs):
        pref = np.full(C, 0.05); pref[fav] = 0.85         # sums to 1.0
        arts = gen.feed(pref, per_user, rng)
        pred = clf.predict(arts)
        hit_pers += (pred == fav).sum()
        # Baseline: non-personalized uniform feed judged against same fav topic.
        base = gen.feed(np.full(C, 1.0 / C), per_user, rng)
        hit_base += (clf.predict(base) == fav).sum()
        if example is None:
            example = (fav, " ".join(inv[i] for i in arts[0]))
    total = n_users * per_user
    pers_rate, base_rate = hit_pers / total, hit_base / total

    print("Topics: %d | vocab: %d | headlines: %d" % (C, V, len(X)))
    print("Topic-judge accuracy (held-out): %.3f  vs majority %.3f" % (acc, majority))
    print("Personalized feed on-topic rate: %.3f" % pers_rate)
    print("Non-personalized baseline rate:  %.3f" % base_rate)
    print("Random baseline (1/C):           %.3f" % (1.0 / C))
    print("Example (topic %d): %s" % example)
    print("Judge beats majority:            %s" % (acc > majority))
    print("Personalization beats baseline:  %s" % (pers_rate > 2 * base_rate))
