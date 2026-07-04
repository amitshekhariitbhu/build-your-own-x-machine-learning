import numpy as np


def edit_distance(a, b):
    # Levenshtein distance via DP: min sub/ins/del edits to turn a into b.
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1,        # deletion
                           dp[i, j - 1] + 1,        # insertion
                           dp[i - 1, j - 1] + cost)  # match / substitution
    return dp[m, n]


class SmartProofreader:
    """Noisy-channel spell corrector: edit-distance candidates ranked by a
    context-aware bigram language model. correct = argmax P(c|prev) - k*dist(w,c)."""

    def __init__(self, max_dist=2, lam=1.6, add_k=0.1):
        self.max_dist = max_dist    # widest edit distance to consider a candidate
        self.lam = lam              # channel penalty per edit (context vs fidelity)
        self.add_k = add_k          # add-k smoothing for the bigram model

    def fit(self, sentences):
        # Learn the vocabulary and unigram/bigram counts from clean text.
        self.vocab = {}
        self.uni = {}
        self.bi = {}
        for toks in sentences:
            prev = "<s>"
            for w in toks:
                self.vocab.setdefault(w, len(self.vocab))
                self.uni[w] = self.uni.get(w, 0) + 1
                self.bi.setdefault(prev, {})
                self.bi[prev][w] = self.bi[prev].get(w, 0) + 1
                prev = w
        self.vocab_list = list(self.vocab)
        self.N = sum(self.uni.values())
        self.V = len(self.vocab)
        self._cand_cache = {}
        return self

    def _log_bigram(self, prev, w):
        # Add-k smoothed P(w|prev); back off to add-k unigram if prev unseen.
        wc = self.uni.get(w, 0)
        if prev in self.bi:
            ctx = self.bi[prev]
            total = sum(ctx.values())
            return np.log((ctx.get(w, 0) + self.add_k) / (total + self.add_k * self.V))
        return np.log((wc + self.add_k) / (self.N + self.add_k * self.V))

    def _candidates(self, word):
        # Vocab words within max_dist edits; always keep the word itself.
        if word in self._cand_cache:
            return self._cand_cache[word]
        cands = {}
        if word in self.vocab:
            cands[word] = 0
        for v in self.vocab_list:
            if abs(len(v) - len(word)) > self.max_dist:
                continue
            d = edit_distance(word, v)
            if d <= self.max_dist and (v not in cands or d < cands[v]):
                cands[v] = d
        if not cands:
            cands[word] = 0          # OOV with no near vocab word: leave as-is
        self._cand_cache[word] = cands
        return cands

    def correct_token(self, prev, word):
        best, best_score = word, -np.inf
        for cand, dist in self._candidates(word).items():
            score = self._log_bigram(prev, cand) - self.lam * dist
            if score > best_score:
                best_score, best = score, cand
        return best

    def correct(self, tokens, use_context=True):
        # Left-to-right decode; feed each corrected word as the next context.
        out, prev = [], "<s>"
        for w in tokens:
            c = self.correct_token(prev if use_context else "<s>", w)
            out.append(c)
            prev = c
        return out


def make_corpus(n=260, seed=0):
    # Synthetic grammatical sentences: templates create stable bigram structure
    # (det->adj->noun->verb...), so context genuinely helps disambiguate typos.
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


def corrupt(tokens, rng, rate=0.3):
    # Inject single-edit typos into ~rate of tokens; return noisy tokens + mask.
    letters = "abcdefghijklmnopqrstuvwxyz"
    out, corrupted = [], []
    for w in tokens:
        if rng.rand() < rate and len(w) >= 2:
            op = rng.choice(["sub", "ins", "del", "swap"])
            i = rng.randint(len(w))
            if op == "sub":
                w2 = w[:i] + rng.choice(list(letters)) + w[i + 1:]
            elif op == "ins":
                w2 = w[:i] + rng.choice(list(letters)) + w[i:]
            elif op == "del":
                w2 = w[:i] + w[i + 1:]
            else:  # swap adjacent characters
                j = min(i, len(w) - 2)
                w2 = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            out.append(w2)
            corrupted.append(w2 != w)
        else:
            out.append(w)
            corrupted.append(False)
    return out, corrupted


def word_accuracy(sents_true, sents_pred):
    tot = hit = 0
    for t, p in zip(sents_true, sents_pred):
        for a, b in zip(t, p):
            tot += 1
            hit += (a == b)
    return hit / tot


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    corpus = make_corpus(n=260, seed=0)
    split = int(0.75 * len(corpus))
    train, test = corpus[:split], corpus[split:]

    pr = SmartProofreader().fit(train)

    # Corrupt the held-out test sentences and try to restore them.
    noisy = [corrupt(s, rng, rate=0.3) for s in test]
    noisy_sents = [n for n, _ in noisy]

    fixed_ctx = [pr.correct(s, use_context=True) for s in noisy_sents]
    fixed_noctx = [pr.correct(s, use_context=False) for s in noisy_sents]

    raw_acc = word_accuracy(test, noisy_sents)     # baseline: leave text as-is
    noctx_acc = word_accuracy(test, fixed_noctx)   # channel only, no context
    ctx_acc = word_accuracy(test, fixed_ctx)       # full smart proofreader

    n_words = sum(len(s) for s in test)
    n_bad = sum(sum(m) for _, m in noisy)

    print("Sentences: %d   Train: %d   Test: %d   Vocab: %d"
          % (len(corpus), len(train), len(test), pr.V))
    print("Test words: %d   Corrupted: %d (%.0f%%)"
          % (n_words, n_bad, 100 * n_bad / n_words))
    print("-" * 58)
    print("Uncorrected (baseline)      word accuracy: %.4f" % raw_acc)
    print("Proofreader (no context)    word accuracy: %.4f" % noctx_acc)
    print("Proofreader (context-aware) word accuracy: %.4f" % ctx_acc)
    print("-" * 58)
    ex = ["teh", "quikc", "brwn", "jmups", "rivr"]
    print("Word-level fixes:  " +
          "   ".join("%s->%s" % (w, pr.correct_token("<s>", w)) for w in ex))
    print("-" * 58)
    print("Proofreader beats uncorrected baseline: %s" % (ctx_acc > raw_acc))
    print("Context helps over no-context:          %s" % (ctx_acc >= noctx_acc))
