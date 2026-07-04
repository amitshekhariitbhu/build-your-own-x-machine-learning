import numpy as np

# Special tokens marking the start / end of a headline for the Markov chain.
START, END = "<s>", "</s>"


def _tok(s):
    return s.split()


def _build_bigram(titles):
    """Bigram (order-1 Markov) language model over a list of token lists.
    Returns word -> (next_words, next_probs) with START/END boundary tokens."""
    trans = {}
    for t in titles:
        seq = [START] + list(t) + [END]
        for a, b in zip(seq, seq[1:]):
            trans.setdefault(a, {})
            trans[a][b] = trans[a].get(b, 0) + 1
    model = {}
    for a, d in trans.items():
        words = list(d.keys())
        counts = np.array([d[w] for w in words], float)
        model[a] = (words, counts / counts.sum())
    return model


def _sample_chain(model, rng, max_len=8):
    """Sample a headline by walking the Markov chain from START to END."""
    if START not in model:
        return []
    out, cur = [], START
    for _ in range(max_len):
        if cur not in model:
            break
        words, probs = model[cur]
        cur = words[rng.choice(len(words), p=probs)]
        if cur == END:
            break
        out.append(cur)
    return out


class NaiveBayesTopic:
    """Multinomial Naive Bayes over bag-of-words to infer an article's topic."""

    def fit(self, docs, labels, n_topics):
        vocab = {}
        for d in docs:
            for w in d:
                vocab.setdefault(w, len(vocab))
        self.vocab, V = vocab, len(vocab)
        counts = np.ones((n_topics, V))            # Laplace smoothing
        prior = np.zeros(n_topics)
        for d, y in zip(docs, labels):
            prior[y] += 1
            for w in d:
                counts[y, vocab[w]] += 1
        self.logprob = np.log(counts / counts.sum(axis=1, keepdims=True))
        self.logprior = np.log(prior / prior.sum())
        return self

    def predict(self, doc):
        score = self.logprior.copy()
        for w in doc:
            j = self.vocab.get(w)
            if j is not None:
                score += self.logprob[:, j]
        return int(np.argmax(score))


class TitleGenerator:
    """Topic-conditioned headline generator. fit() learns (a) a Naive Bayes
    topic model over article bodies and (b) one bigram headline chain PER topic.
    generate(body) infers the topic from the body, then samples a headline from
    that topic's chain -- so the title matches the article's subject."""

    def fit(self, bodies, titles, topics, n_topics, seed=1):
        self.n_topics = n_topics
        self.rng = np.random.RandomState(seed)
        self.nb = NaiveBayesTopic().fit(bodies, topics, n_topics)
        # Per-topic chain (ours) and a topic-agnostic pooled chain (baseline).
        self.chains = [_build_bigram([ti for ti, tp in zip(titles, topics) if tp == k])
                       for k in range(n_topics)]
        self.global_chain = _build_bigram(titles)
        self.title_vocab = sorted({w for t in titles for w in t})
        return self

    def generate(self, body):
        k = self.nb.predict(body)
        return _sample_chain(self.chains[k], self.rng)

    def generate_global(self, _body):
        return _sample_chain(self.global_chain, self.rng)   # ignores the topic

    def generate_random(self, _body):
        L = self.rng.randint(3, 7)
        idx = self.rng.choice(len(self.title_vocab), L)
        return [self.title_vocab[i] for i in idx]


def rouge1_f1(gen, ref):
    """Unigram overlap F1 between a generated and a reference headline."""
    g, r = set(gen), set(ref)
    if not g or not r:
        return 0.0
    ov = len(g & r)
    prec, rec = ov / len(g), ov / len(r)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


# ---- Synthetic corpus: articles whose body words and headline share a topic --
_TOPIC_WORDS = {
    "space":   "nasa rocket mars orbit launch mission satellite moon galaxy "
               "telescope rover planet astronaut cosmic probe".split(),
    "cooking": "recipe oven bake flavor sauce kitchen spice dish chef dough "
               "roast dinner bread cake garlic".split(),
    "finance": "market stock invest profit bank trade revenue price economy "
               "fund rate investor growth gold shares".split(),
    "sports":  "team match score player coach league goal championship season "
               "final playoff victory win striker defender".split(),
}
_SEED_TITLES = {
    "space":   ["nasa launches new mars mission", "rocket reaches orbit after launch",
                "telescope spots a distant galaxy", "moon mission deploys new satellite",
                "mars rover explores the red planet"],
    "cooking": ["easy recipe for perfect bread", "bake the best chocolate cake",
                "chef shares a secret sauce recipe", "spice up your kitchen dinner",
                "oven roast garlic for more flavor"],
    "finance": ["stock market hits a record high", "invest wisely in growth fund",
                "bank raises the interest rate again", "profit rises as revenue grows",
                "traders watch the price of gold"],
    "sports":  ["team wins the championship final", "star player scores winning goal",
                "coach leads league title season", "season ends with playoff victory",
                "striker scores as team win match"],
}
_NOISE = "report study today world people time year update said review".split()


def make_corpus(n_docs, topic_names, seed=0):
    rng = np.random.RandomState(seed)
    true_chain = {k: _build_bigram([_tok(s) for s in _SEED_TITLES[name]])
                  for k, name in enumerate(topic_names)}
    bodies, titles, topics = [], [], []
    for _ in range(n_docs):
        k = rng.randint(len(topic_names))
        tw = _TOPIC_WORDS[topic_names[k]]
        body = [tw[i] for i in rng.randint(len(tw), size=28)]      # topic content
        body += [_NOISE[i] for i in rng.randint(len(_NOISE), size=12)]  # filler
        bodies.append(body)
        titles.append(_sample_chain(true_chain[k], rng))
        topics.append(k)
    return bodies, titles, topics


if __name__ == "__main__":
    np.random.seed(0)
    names = list(_TOPIC_WORDS)
    bodies, titles, topics = make_corpus(700, names, seed=0)

    n_tr = 500
    gen = TitleGenerator().fit(bodies[:n_tr], titles[:n_tr], topics[:n_tr], len(names))

    te_bodies, te_titles, te_topics = bodies[n_tr:], titles[n_tr:], topics[n_tr:]
    r_ours = r_glob = r_rand = 0.0
    title_topic_acc = 0.0
    samples = []
    for b, ref, tp in zip(te_bodies, te_titles, te_topics):
        g = gen.generate(b)
        r_ours += rouge1_f1(g, ref)
        r_glob += rouge1_f1(gen.generate_global(b), ref)
        r_rand += rouge1_f1(gen.generate_random(b), ref)
        title_topic_acc += (gen.nb.predict(g) == tp)   # does the title read on-topic?
        if len(samples) < 4:
            samples.append((names[tp], " ".join(ref), " ".join(g)))
    m = len(te_titles)
    r_ours, r_glob, r_rand = r_ours / m, r_glob / m, r_rand / m
    title_topic_acc /= m

    print("Sample generations (topic | reference -> generated):")
    for tp, ref, g in samples:
        print("  [%-7s] %-32s -> %s" % (tp, ref, g))
    print()
    print("Held-out articles              :", m)
    print("Random-title      ROUGE-1 F1   :", round(r_rand, 3))
    print("Global-chain      ROUGE-1 F1   :", round(r_glob, 3), "(no topic conditioning)")
    print("Topic-conditioned ROUGE-1 F1   :", round(r_ours, 3), "(ours)")
    print("Generated-title topic accuracy :", round(title_topic_acc, 3),
          "vs chance", round(1.0 / len(names), 3))
    print()
    ok = (r_ours > 1.5 * r_glob and r_ours > 3 * r_rand
          and title_topic_acc > 0.85)
    print("RESULT:", "PASS - topic-conditioned titles beat every baseline" if ok else "FAIL")
