import numpy as np

# ---- QWERTY geometry: nearby keys are more likely to be mistyped -------------
_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_POS = {ch: (r, c) for r, row in enumerate(_ROWS) for c, ch in enumerate(row)}
_LETTERS = "abcdefghijklmnopqrstuvwxyz"

def _euclid(a, b):
    (r1, c1), (r2, c2) = _POS[a], _POS[b]
    return ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5

# Adjacency map (neighbours within ~one key) used to plant realistic typos.
_NB = {a: [b for b in _LETTERS if b != a and _euclid(a, b) <= 1.5] or [a]
       for a in _LETTERS}

def _sub_cost(a, b, keyboard):
    if a == b:
        return 0.0
    if not keyboard:
        return 1.0
    return min(1.0, _euclid(a, b) / 4.0)   # adjacent keys ~0.25, far keys ~1.0

def _edit_dist(a, b, keyboard=True):
    """Damerau-Levenshtein with keyboard-weighted substitution (noisy channel)."""
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min(d[i-1, j-1] + _sub_cost(a[i-1], b[j-1], keyboard),
                       d[i-1, j] + 1.0, d[i, j-1] + 1.0)
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                best = min(best, d[i-2, j-2] + 0.8)   # transposition
            d[i, j] = best
    return d[n, m]


class AutocorrectKeyboard:
    """Noisy-channel spell corrector: argmax_c  logP(c) - lam * dist(typed, c),
    where dist is keyboard-aware so near-key typos are cheap to explain."""

    def __init__(self, lam=6.0):
        self.lam = lam
        self.logp = {}     # word -> log prior probability (language model)
        self.words = set()

    def fit(self, freqs):
        total = sum(freqs.values())
        self.words = set(freqs)
        self.logp = {w: np.log(c / total) for w, c in freqs.items()}
        return self

    def _edits1(self, w):
        sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        dele = [L + R[1:] for L, R in sp if R]
        trans = [L + R[1] + R[0] + R[2:] for L, R in sp if len(R) > 1]
        repl = [L + c + R[1:] for L, R in sp if R for c in _LETTERS]
        ins = [L + c + R for L, R in sp for c in _LETTERS]
        return set(dele + trans + repl + ins)

    def _known(self, cs):
        return {w for w in cs if w in self.words}

    def candidates(self, w):
        if w in self.words:
            return {w}
        c1 = self._known(self._edits1(w))
        if c1:
            return c1
        c2 = set()                                   # distance-2 fallback
        for e in self._edits1(w):
            c2 |= self._known(self._edits1(e))
        return c2

    def correct(self, w, mode="keyboard"):
        cands = self.candidates(w)
        if not cands:
            return w
        best, best_s = w, -np.inf
        for c in cands:
            prior = self.logp.get(c, -20.0)
            if mode == "freq":                       # ignore how the typo looks
                s = prior
            else:                                    # channel: keyboard vs plain
                s = prior - self.lam * _edit_dist(w, c, keyboard=(mode == "keyboard"))
            if s > best_s:
                best_s, best = s, c
        return best


def _make_typo(w, n):
    """Apply n keyboard-realistic corruptions (adjacent sub / transpose / ins / del)."""
    for _ in range(n):
        op = np.random.choice(["sub", "trans", "ins", "del"], p=[.75, .1, .075, .075])
        i = np.random.randint(len(w))
        if op == "trans" and len(w) > 1:
            i = np.random.randint(len(w) - 1)
            w = w[:i] + w[i+1] + w[i] + w[i+2:]
        elif op == "ins":
            w = w[:i] + np.random.choice(_NB[w[i]]) + w[i:]
        elif op == "del" and len(w) > 2:
            w = w[:i] + w[i+1:]
        else:                                        # substitution (adjacent key)
            w = w[:i] + np.random.choice(_NB[w[i]]) + w[i+1:]
    return w


# Example usage
if __name__ == "__main__":
    np.random.seed(0)

    # Planted structure: a lexicon full of one-letter neighbours (cat/car/can...)
    # so correction is ambiguous and keyboard geometry is needed to disambiguate.
    words = ["cat", "car", "can", "cap", "cab", "bat", "hat", "mat", "rat", "sat",
             "bar", "bag", "big", "bit", "bin", "ban", "pan", "pen", "pin", "pit",
             "sit", "sip", "tip", "top", "tap", "map", "mad", "bad", "bed", "bee",
             "sea", "tea", "ten", "hen", "her", "care", "core", "cord", "word",
             "ward", "cart", "part", "park", "dark", "bark", "back", "pack", "rack",
             "lock", "lick", "like", "lake", "bake", "cake", "make", "more", "mode",
             "code", "node", "line", "lime", "time", "tile", "mile", "mild", "wild",
             "will", "wall", "ball", "tall", "bell", "belt", "boot", "book", "look",
             "cook", "cool", "tool", "pool", "door"]

    # Zipfian frequencies (language-model prior); rank order randomised.
    ranks = np.random.permutation(len(words))
    freqs = {w: int(3000 / (r + 1)) + 1 for w, r in zip(words, ranks)}
    p_word = np.array([freqs[w] for w in words], float)
    p_word /= p_word.sum()
    top_word = words[int(np.argmax(p_word))]

    clf = AutocorrectKeyboard().fit(freqs)

    # Held-out test: sample a word by frequency, corrupt it into a NON-word typo.
    tests = []
    while len(tests) < 300:
        true = words[np.random.choice(len(words), p=p_word)]
        typo = _make_typo(true, np.random.choice([1, 2], p=[.7, .3]))
        if typo not in clf.words and typo != true:   # keep genuine misspellings
            tests.append((typo, true))

    acc_kbd  = np.mean([clf.correct(t, "keyboard") == y for t, y in tests])
    acc_plain= np.mean([clf.correct(t, "plain")    == y for t, y in tests])
    acc_freq = np.mean([clf.correct(t, "freq")     == y for t, y in tests])
    acc_noop = np.mean([t == y for t, y in tests])          # no autocorrect
    acc_maj  = np.mean([top_word == y for t, y in tests])   # always guess top word

    print("Sample corrections (typo -> fix / truth):")
    for t, y in tests[:6]:
        print("  %-6s -> %-6s (%s)" % (t, clf.correct(t), y))
    print()
    print("Test misspellings          :", len(tests))
    print("No-autocorrect  accuracy   :", round(float(acc_noop), 3))
    print("Majority-word   accuracy   :", round(float(acc_maj), 3))
    print("Freq-only       accuracy   :", round(float(acc_freq), 3))
    print("Plain-channel   accuracy   :", round(float(acc_plain), 3))
    print("Keyboard-aware  accuracy   :", round(float(acc_kbd), 3))
    print()
    ok = (acc_kbd > acc_plain and acc_kbd > acc_freq and acc_kbd > 0.7
          and acc_kbd > 3 * max(acc_noop, acc_maj))
    print("RESULT:", "PASS - keyboard model beats every baseline" if ok else "FAIL")
