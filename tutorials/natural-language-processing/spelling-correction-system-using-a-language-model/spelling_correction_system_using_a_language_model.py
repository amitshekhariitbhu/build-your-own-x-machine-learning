import numpy as np

LETTERS = "abcdefghijklmnopqrstuvwxyz"


class LanguageModel:
    """Interpolated bigram language model with add-k smoothing, from scratch.

    Learns unigram and bigram counts from clean sentences. log_bi(prev, w)
    gives log P(w | prev) as a mix of the smoothed bigram and unigram laws,
    so a strong context (prev) drives the estimate but an unseen context
    still backs off to the word's own frequency instead of zero."""

    def __init__(self, add_k=0.05, lam=0.7):
        self.add_k = add_k    # add-k smoothing mass
        self.lam = lam        # weight on the bigram term vs unigram backoff

    def fit(self, sentences):
        self.uni, self.bi, self.vocab = {}, {}, set()
        for toks in sentences:
            prev = "<s>"
            for w in toks:
                self.vocab.add(w)
                self.uni[w] = self.uni.get(w, 0) + 1
                self.bi.setdefault(prev, {})
                self.bi[prev][w] = self.bi[prev].get(w, 0) + 1
                prev = w
        self.N = sum(self.uni.values())
        self.V = len(self.vocab)
        return self

    def _p_uni(self, w):
        return (self.uni.get(w, 0) + self.add_k) / (self.N + self.add_k * self.V)

    def log_bi(self, prev, w):
        p_uni = self._p_uni(w)
        ctx = self.bi.get(prev)
        if not ctx:
            return float(np.log(p_uni))
        total = sum(ctx.values())
        p_bi = (ctx.get(w, 0) + self.add_k) / (total + self.add_k * self.V)
        return float(np.log(self.lam * p_bi + (1 - self.lam) * p_uni))


class SpellingCorrector:
    """Noisy-channel speller: argmax_c  log P(c|prev) + log P(typo|c).

    Candidates come from Norvig-style edits (delete/transpose/replace/insert)
    restricted to known words -- distance 1 first, distance 2 only if needed.
    A whole sentence is decoded with Viterbi over the bigram language model,
    so context can even repair a valid-but-wrong word (real-word error)."""

    def __init__(self, theta=4.0, add_k=0.05, lam=0.7):
        self.theta = theta    # channel penalty per edit (fidelity vs context)
        self.lm = LanguageModel(add_k=add_k, lam=lam)

    def fit(self, sentences):
        self.lm.fit(sentences)
        self.vocab = self.lm.vocab
        self._cache = {}
        return self

    def _edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        out = set()
        out.update(L + R[1:] for L, R in splits if R)                       # delete
        out.update(L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1)  # swap
        out.update(L + c + R[1:] for L, R in splits if R for c in LETTERS)  # replace
        out.update(L + c + R for L, R in splits for c in LETTERS)           # insert
        return out

    def candidates(self, word):
        # dict {candidate: edit_distance}. Keep the word itself (dist 0) plus
        # known neighbours so the LM may override a real-word error.
        if word in self._cache:
            return self._cache[word]
        cands = {}
        if word in self.vocab:
            cands[word] = 0
        for w in self._edits1(word):
            if w in self.vocab and w not in cands:
                cands[w] = 1
        if not cands:  # nothing within one edit: widen to distance 2
            for w1 in self._edits1(word):
                for w2 in self._edits1(w1):
                    if w2 in self.vocab and w2 not in cands:
                        cands[w2] = 2
        if not cands:
            cands[word] = 0        # OOV, no near vocab word: leave unchanged
        self._cache[word] = cands
        return cands

    def _channel(self, dist):
        return -self.theta * dist  # log P(observed | candidate): decays per edit

    def correct(self, tokens, use_context=True):
        # Viterbi decode: state = a candidate at each position, transition score
        # = bigram log-prob, emission = channel score. use_context=False falls
        # back to a per-word argmax (unigram LM only) -- the no-context baseline.
        cand_sets = [self.candidates(w) for w in tokens]
        if not tokens:
            return []
        score, back = [], []
        first = {}
        for c, d in cand_sets[0].items():
            first[c] = self.lm.log_bi("<s>", c) + self._channel(d)
        score.append(first)
        back.append({c: None for c in first})
        for i in range(1, len(tokens)):
            cur, bp = {}, {}
            for c, d in cand_sets[i].items():
                em = self._channel(d)
                if use_context:
                    best_p, best_s = None, -np.inf
                    for p, ps in score[i - 1].items():
                        s = ps + self.lm.log_bi(p, c) + em
                        if s > best_s:
                            best_s, best_p = s, p
                else:  # no context: each word scored on its own via unigram LM
                    best_p = max(score[i - 1], key=score[i - 1].get)
                    best_s = float(np.log(self.lm._p_uni(c))) + em
                cur[c], bp[c] = best_s, best_p
            score.append(cur)
            back.append(bp)
        last = max(score[-1], key=score[-1].get)
        out = [last]
        for i in range(len(tokens) - 1, 0, -1):
            out.append(back[i][out[-1]])
        return out[::-1]


def make_corpus(n=300, seed=0):
    # Templated grammatical sentences give stable det->adj->noun->verb bigram
    # structure, so context genuinely disambiguates typos.
    rng = np.random.RandomState(seed)
    det = ["the", "a"]
    adj = ["quick", "lazy", "happy", "brown", "small", "bright", "clever"]
    noun = ["cat", "dog", "fox", "bird", "child", "teacher", "student", "river"]
    verb = ["runs", "jumps", "reads", "sees", "eats", "finds", "chases"]
    prep = ["over", "near", "under", "beside"]
    sents = []
    for _ in range(n):
        s = [rng.choice(det), rng.choice(adj), rng.choice(noun), rng.choice(verb)]
        if rng.rand() < 0.7:
            s += [rng.choice(prep), rng.choice(det), rng.choice(adj), rng.choice(noun)]
        sents.append(s)
    return sents


def corrupt(tokens, rng, rate=0.35):
    # Inject single-edit typos into ~rate of tokens; return noisy tokens + mask.
    out, mask = [], []
    for w in tokens:
        if rng.rand() < rate and len(w) >= 2:
            op = rng.choice(["sub", "ins", "del", "swap"])
            i = rng.randint(len(w))
            if op == "sub":
                w2 = w[:i] + rng.choice(list(LETTERS)) + w[i + 1:]
            elif op == "ins":
                w2 = w[:i] + rng.choice(list(LETTERS)) + w[i:]
            elif op == "del":
                w2 = w[:i] + w[i + 1:]
            else:
                j = min(i, len(w) - 2)
                w2 = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            out.append(w2)
            mask.append(w2 != w)
        else:
            out.append(w)
            mask.append(False)
    return out, mask


def word_accuracy(truth, pred):
    tot = hit = 0
    for t, p in zip(truth, pred):
        for a, b in zip(t, p):
            tot += 1
            hit += (a == b)
    return hit / tot


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    corpus = make_corpus(n=300, seed=0)
    split = int(0.75 * len(corpus))
    train, test = corpus[:split], corpus[split:]

    sc = SpellingCorrector(theta=4.0).fit(train)

    noisy = [corrupt(s, rng, rate=0.35) for s in test]
    noisy_sents = [n for n, _ in noisy]

    fixed_ctx = [sc.correct(s, use_context=True) for s in noisy_sents]
    fixed_noctx = [sc.correct(s, use_context=False) for s in noisy_sents]

    raw_acc = word_accuracy(test, noisy_sents)     # baseline: do nothing
    noctx_acc = word_accuracy(test, fixed_noctx)   # channel + unigram LM
    ctx_acc = word_accuracy(test, fixed_ctx)       # full bigram Viterbi speller

    n_words = sum(len(s) for s in test)
    n_bad = sum(sum(m) for _, m in noisy)

    print("Sentences: %d   Train: %d   Test: %d   Vocab: %d"
          % (len(corpus), len(train), len(test), sc.lm.V))
    print("Test words: %d   Corrupted: %d (%.0f%%)"
          % (n_words, n_bad, 100 * n_bad / n_words))
    print("-" * 60)
    print("Uncorrected (baseline)       word accuracy: %.4f" % raw_acc)
    print("Speller (no context, unigram) word accuracy: %.4f" % noctx_acc)
    print("Speller (context, bigram LM)  word accuracy: %.4f" % ctx_acc)
    print("-" * 60)
    ex = ["teh", "quikc", "brwn", "jmups", "rivr", "chld"]
    print("Isolated fixes:  " +
          "   ".join("%s->%s" % (w, list(sc.candidates(w))[0]) for w in ex))
    print("-" * 60)
    print("Speller beats uncorrected baseline: %s" % (ctx_acc > raw_acc))
    print("Language model context helps:       %s" % (ctx_acc >= noctx_acc))
