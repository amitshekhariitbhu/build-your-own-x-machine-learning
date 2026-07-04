import numpy as np
from collections import Counter


class BPETokenizer:
    """Byte-Pair Encoding tokenizer from scratch.

    Train: start from characters, then repeatedly merge the most frequent
    adjacent symbol pair into a new symbol. The learned merge list turns
    frequent character sequences (morphemes) into single subword tokens, so
    text is encoded in far fewer tokens than a character tokenizer while
    still decoding back exactly."""

    def __init__(self, num_merges=180):
        self.num_merges = num_merges
        self.merges = []          # ordered merges learned during fit
        self.vocab = {}           # token string -> id
        self.id2tok = {}          # id -> token string

    @staticmethod
    def _apply(symbols, a, b):
        # Merge every adjacent (a, b) in a symbol list into the token a+b.
        merged, out, i = a + b, [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out

    def fit(self, corpus):
        # corpus: list of words. Each unique word -> chars + end marker '</w>'.
        word_freq = Counter(corpus)
        words = {tuple(w) + ("</w>",): c for w, c in word_freq.items()}

        for _ in range(self.num_merges):
            # Count adjacent symbol pairs across all words, weighted by count.
            pairs = Counter()
            for sym, freq in words.items():
                for i in range(len(sym) - 1):
                    pairs[(sym[i], sym[i + 1])] += freq
            if not pairs:
                break
            (a, b), best = pairs.most_common(1)[0]
            if best < 2:                       # nothing recurring left to merge
                break
            self.merges.append((a, b))
            words = {tuple(self._apply(list(s), a, b)): f for s, f in words.items()}

        # Vocab = every base character seen + '</w>' + every merged token.
        symbols = {"</w>"}
        for w in word_freq:
            symbols.update(w)
        for a, b in self.merges:
            symbols.add(a + b)
        self.vocab = {s: i for i, s in enumerate(sorted(symbols))}
        self.id2tok = {i: s for s, i in self.vocab.items()}
        return self

    def _tokenize_word(self, word):
        symbols = list(word) + ["</w>"]
        for a, b in self.merges:           # apply merges in the order learned
            symbols = self._apply(symbols, a, b)
        return symbols

    def encode(self, words):
        # words: list of word strings -> flat list of token ids.
        ids = []
        for w in words:
            for tok in self._tokenize_word(w):
                ids.append(self.vocab[tok])
        return ids

    def decode(self, ids):
        # Concatenate token strings; '</w>' marks word boundaries.
        text = "".join(self.id2tok[i] for i in ids)
        return text.replace("</w>", " ").strip()


def make_words(prefixes, roots, suffixes, n, rng):
    # Planted structure: words = prefix + root + suffix, so subword units
    # (roots, 'ing', 'tion', ...) recur constantly for BPE to discover.
    return [rng.choice(prefixes) + rng.choice(roots) + rng.choice(suffixes)
            for _ in range(n)]


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    np.random.seed(0)

    prefixes = ["", "", "re", "un", "pre", "dis", "over"]
    roots = ["play", "work", "read", "walk", "talk", "jump",
             "call", "help", "move", "turn", "build", "form"]
    suffixes = ["", "ing", "ed", "s", "er", "ment", "able", "tion", "ful"]

    train = make_words(prefixes, roots, suffixes, 4000, rng)   # training corpus
    test = make_words(prefixes, roots, suffixes, 400, rng)     # held-out words

    bpe = BPETokenizer(num_merges=180).fit(train)

    # --- Correctness signal 1: exact round-trip on held-out text -----------
    exact = 0
    for w in test:
        exact += (bpe.decode(bpe.encode([w])) == w)
    roundtrip = exact / len(test)

    # --- Correctness signal 2: compression vs a character tokenizer --------
    # Baseline char tokenizer: each char + '</w>' is its own token.
    base_tokens = sum(len(w) + 1 for w in test)
    bpe_tokens = len(bpe.encode(test))
    ratio = base_tokens / bpe_tokens
    base_tpw = base_tokens / len(test)
    bpe_tpw = bpe_tokens / len(test)

    learned = ["".join(m) for m in bpe.merges[:12]]

    print("Train words: %d  Held-out words: %d  Merges: %d  Vocab: %d"
          % (len(train), len(test), len(bpe.merges), len(bpe.vocab)))
    print("Sample learned subwords:", learned)
    print("Round-trip exact reconstruction: %.3f  (target 1.000)" % roundtrip)
    print("Tokens/word  char baseline: %.2f" % base_tpw)
    print("Tokens/word  BPE:           %.2f" % bpe_tpw)
    print("Compression ratio (BPE vs char): %.2fx  (baseline 1.00x)" % ratio)
    ok = roundtrip == 1.0 and ratio > 1.5
    print("SUCCESS" if ok else "FAIL")
