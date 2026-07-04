import numpy as np

# Language Translator from scratch (statistical machine translation).
#
# We synthesize a parallel corpus of (foreign, English) sentence pairs from a
# tiny grammar and a HIDDEN bilingual dictionary. The foreign side reorders each
# noun phrase (adjective AFTER the noun, Romance-style) so translation is not a
# trivial position-for-position copy -- word order must be repaired too.
#
# Everything is hand-rolled:
#   1) IBM Model 1 word alignment learned by EM -- recovers t(english | foreign),
#      i.e. the hidden dictionary, from sentence pairs alone (no dictionary given).
#   2) A bigram language model over the English side, used to greedily reorder the
#      word-for-word gloss into fluent English via adjacent swaps.
#
# Held-out report: order-independent word-translation accuracy and ordered
# position accuracy, both vs a majority-word and a random-word baseline.

# Hidden bilingual dictionary (English -> foreign). The learner never sees this.
ENG2FRN = {
    "the": "le", "a": "un",
    "small": "pequen", "happy": "feliz", "red": "roja", "clever": "lista", "quiet": "silen",
    "cat": "gato", "dog": "perro", "child": "nino", "robot": "robo", "teacher": "maest", "river": "rio",
    "sees": "mira", "likes": "ama", "chases": "persi", "paints": "pinta", "carries": "lleva",
}
DETS = ["the", "a"]
ADJS = ["small", "happy", "red", "clever", "quiet"]
NOUNS = ["cat", "dog", "child", "robot", "teacher", "river"]
VERBS = ["sees", "likes", "chases", "paints", "carries"]


def make_pair(rng):
    # Build one English sentence and its reordered foreign translation.
    def np_phrase():
        det, noun = rng.choice(DETS), rng.choice(NOUNS)
        adj = rng.choice(ADJS) if rng.random() < 0.7 else None
        eng = [det] + ([adj] if adj else []) + [noun]          # det (adj) noun
        frn = [ENG2FRN[det], ENG2FRN[noun]] + ([ENG2FRN[adj]] if adj else [])  # det noun (adj)
        return eng, frn
    se, sf = np_phrase()                                        # subject
    oe, of = np_phrase()                                        # object
    verb = rng.choice(VERBS)
    eng = se + [verb] + oe
    frn = sf + [ENG2FRN[verb]] + of
    return eng, frn


class Model1Translator:
    """IBM Model 1 lexical translation (EM) + bigram-LM reordering."""

    NULL = "<null>"

    def fit(self, pairs, em_iters=15):
        eng_words = sorted({w for e, _ in pairs for w in e})
        frn_words = [self.NULL] + sorted({w for _, f in pairs for w in f})
        self.ei = {w: i for i, w in enumerate(eng_words)}       # english -> index
        self.fi = {w: i for i, w in enumerate(frn_words)}       # foreign -> index
        self.eng_words, self.frn_words = eng_words, frn_words
        E, F = len(eng_words), len(frn_words)

        # Pre-index pairs; every foreign side carries a NULL word (index 0).
        idx = [(np.array([self.ei[w] for w in e]),
                np.array([0] + [self.fi[w] for w in f])) for e, f in pairs]

        # t(e|f): columns (over english, for a fixed foreign word) sum to 1.
        t = np.full((E, F), 1.0 / E)
        for _ in range(em_iters):
            count = np.zeros((E, F))
            total = np.zeros(F)
            for ea, fa in idx:
                sub = t[np.ix_(ea, fa)]                          # (m, l)
                delta = sub / sub.sum(axis=1, keepdims=True)     # expected alignments
                er = np.repeat(ea, len(fa))
                fr = np.tile(fa, len(ea))
                np.add.at(count, (er, fr), delta.ravel())
                np.add.at(total, fr, delta.ravel())
            t = count / np.maximum(total, 1e-12)                 # M-step
        self.t = t

        # Bigram language model over the English side (add-1 smoothed logs).
        self.lm_vocab = eng_words + ["<s>", "</s>"]
        self.li = {w: i for i, w in enumerate(self.lm_vocab)}
        V = len(self.lm_vocab)
        bg = np.ones((V, V))
        for e, _ in pairs:
            seq = ["<s>"] + e + ["</s>"]
            for a, b in zip(seq[:-1], seq[1:]):
                bg[self.li[a], self.li[b]] += 1.0
        self.logbg = np.log(bg / bg.sum(axis=1, keepdims=True))
        return self

    def gloss(self, frn):
        # Word-for-word: pick the most likely English word for each foreign word.
        return [self.eng_words[self.t[:, self.fi[w]].argmax()] for w in frn]

    def _score(self, toks):
        seq = ["<s>"] + toks + ["</s>"]
        return sum(self.logbg[self.li[a], self.li[b]] for a, b in zip(seq[:-1], seq[1:]))

    def reorder(self, toks):
        # Greedy hill-climb over adjacent swaps to maximise LM fluency.
        best, bs = list(toks), self._score(toks)
        improved = True
        while improved:
            improved = False
            for i in range(len(best) - 1):
                cand = best[:i] + [best[i + 1], best[i]] + best[i + 2:]
                cs = self._score(cand)
                if cs > bs + 1e-9:
                    best, bs, improved = cand, cs, True
        return best

    def translate(self, frn):
        return self.reorder(self.gloss(frn))


def overlap(pred, ref):
    # Order-independent multiset match / reference length (BLEU-1 style).
    ref = list(ref)
    hit = 0
    for w in pred:
        if w in ref:
            ref.remove(w); hit += 1
    return hit / max(len(ref) + hit, 1)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    np.random.seed(0)

    pairs = [make_pair(rng) for _ in range(600)]
    train, test = pairs[:480], pairs[480:]

    model = Model1Translator().fit(train)

    # Dictionary recovery: does argmax t(e|f) equal the true translation?
    rec = np.mean([model.eng_words[model.t[:, model.fi[f]].argmax()] == e
                   for e, f in ENG2FRN.items()])

    majority = max((w for e, _ in train for w in e),
                   key=[w for e, _ in train for w in e].count)
    eng_vocab = model.eng_words

    word_acc = maj_acc = rnd_acc = 0.0
    pos_hit = pos_tot = gloss_hit = 0
    for eng, frn in test:
        pred = model.translate(frn)
        word_acc += overlap(model.gloss(frn), eng)                 # lexical quality
        maj_acc += overlap([majority] * len(eng), eng)             # majority baseline
        rnd_acc += overlap(list(rng.choice(eng_vocab, len(eng))), eng)  # random baseline
        gloss_hit += sum(a == b for a, b in zip(model.gloss(frn), eng))  # before reorder
        pos_hit += sum(a == b for a, b in zip(pred, eng))          # after reorder
        pos_tot += len(eng)
    n = len(test)

    print("=== Language Translator (IBM Model 1 + bigram LM reordering) ===")
    print("train pairs: {}  test pairs: {}".format(len(train), len(test)))
    print("hidden dictionary recovered:          {:.1f}%".format(100 * rec))
    print("-- word-translation accuracy (order-independent) --")
    print("  random-word baseline:               {:.1f}%".format(100 * rnd_acc / n))
    print("  majority-word baseline ('{}'):     {:.1f}%".format(majority, 100 * maj_acc / n))
    print("  Model 1 translator:                 {:.1f}%".format(100 * word_acc / n))
    print("-- ordered position accuracy --")
    print("  word-for-word gloss (no reorder):   {:.1f}%".format(100 * gloss_hit / pos_tot))
    print("  after LM reordering:                {:.1f}%".format(100 * pos_hit / pos_tot))
    ok = (word_acc / n > 0.9) and (word_acc / n > 3 * maj_acc / n)
    print("RESULT:", "PASS -- translator beats baselines" if ok else "FAIL")
    e, f = test[0]
    print("example  FRN:", " ".join(f))
    print("         ->  ", " ".join(model.translate(f)), " (ref:", " ".join(e) + ")")
