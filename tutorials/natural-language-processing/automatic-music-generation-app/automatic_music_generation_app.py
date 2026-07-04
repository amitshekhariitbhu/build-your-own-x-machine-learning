import numpy as np

# Automatic Music Generation, from scratch.
# We treat music as language: each NOTE is a "word" and a melody is a
# "sentence". We learn an order-k n-gram (Markov) model over notes, then
# SAMPLE new melodies from it. Correctness is proven two ways on held-out
# songs: (1) next-note prediction accuracy vs a majority-note baseline,
# (2) perplexity of the n-gram vs a unigram baseline.

# ---- Vocabulary: two octaves of a C-major scale plus a rest ----
NOTES = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
         'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5', 'REST']
NOTE2IDX = {n: i for i, n in enumerate(NOTES)}
V = len(NOTES)

# ---- Motif bank: short musical phrases = the planted latent structure ----
# Real melodies are locally predictable (arpeggios, scale runs, cadences).
# Chaining these motifs gives strong sequential regularity for the model
# to recover -- NOT structureless noise.
MOTIFS = [
    ['C4', 'E4', 'G4', 'C5'],                     # C major arpeggio up
    ['C5', 'G4', 'E4', 'C4'],                     # C major arpeggio down
    ['G4', 'B4', 'D5', 'G5'],                     # G major arpeggio up
    ['C4', 'D4', 'E4', 'F4', 'G4'],               # ascending scale run
    ['G5', 'F5', 'E5', 'D5', 'C5'],               # descending scale run
    ['E4', 'G4', 'C5', 'G4'],                     # neighbor figure
    ['A4', 'C5', 'E5', 'C5'],                     # A minor arpeggio
    ['F4', 'A4', 'C5', 'A4'],                     # F major arpeggio
    ['C4', 'C4', 'G4', 'G4', 'A4', 'A4', 'G4'],   # twinkle-like phrase
    ['E4', 'D4', 'C4', 'REST'],                   # cadence + rest
]


def make_song(rng, n_motifs):
    """Build one melody by chaining random motifs into a note-index sequence."""
    seq = []
    for _ in range(n_motifs):
        motif = MOTIFS[rng.randint(len(MOTIFS))]
        seq.extend(NOTE2IDX[n] for n in motif)
    return seq


class NGramMusic:
    """Order-k Markov model over notes with add-alpha smoothing."""

    def __init__(self, order=2, alpha=0.1):
        self.order = order
        self.alpha = alpha

    def fit(self, songs):
        k = self.order
        self.counts = {}                 # context tuple -> V-length count vector
        self.uni = np.zeros(V)           # unigram counts (fallback distribution)
        for s in songs:
            for n in s:
                self.uni[n] += 1
            for i in range(len(s) - k):
                ctx = tuple(s[i:i + k])
                self.counts.setdefault(ctx, np.zeros(V))[s[i + k]] += 1
        return self

    def dist(self, ctx):
        """Smoothed next-note distribution given a context; unigram fallback."""
        base = self.counts.get(tuple(ctx))
        base = self.uni if base is None else base
        p = base + self.alpha
        return p / p.sum()

    def predict_next(self, ctx):
        return int(np.argmax(self.dist(ctx)))

    def generate(self, seed, length, rng):
        seq = list(seed)
        for _ in range(length):
            p = self.dist(seq[-self.order:] if self.order else [])
            seq.append(int(rng.choice(V, p=p)))
        return seq

    def perplexity(self, songs):
        k, ll, n = self.order, 0.0, 0
        for s in songs:
            for i in range(len(s) - k):
                ll += np.log(self.dist(s[i:i + k])[s[i + k]])
                n += 1
        return float(np.exp(-ll / n))


def next_note_accuracy(model, songs, fixed=None):
    """Accuracy of predicting each held-out note; if `fixed` set, always guess it."""
    k, correct, tot = model.order, 0, 0
    for s in songs:
        for i in range(len(s) - k):
            pred = fixed if fixed is not None else model.predict_next(s[i:i + k])
            correct += (pred == s[i + k])
            tot += 1
    return correct / tot


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    # Synthetic corpus of melodies, held-out split.
    songs = [make_song(rng, rng.randint(6, 12)) for _ in range(140)]
    train, test = songs[:110], songs[110:]

    model = NGramMusic(order=2, alpha=0.1).fit(train)
    unigram = NGramMusic(order=0, alpha=0.1).fit(train)  # baseline model

    # Correctness signal 1: held-out next-note prediction accuracy.
    acc = next_note_accuracy(model, test)
    majority = int(np.argmax(model.uni))                 # most frequent note
    base_acc = next_note_accuracy(model, test, fixed=majority)

    # Correctness signal 2: held-out perplexity (lower is better).
    ng_ppl = model.perplexity(test)
    uni_ppl = unigram.perplexity(test)

    print("=== Automatic Music Generation (order-2 n-gram over notes) ===")
    print("vocabulary size: %d notes, train songs: %d, test songs: %d"
          % (V, len(train), len(test)))
    print()
    print("Next-note prediction accuracy (held-out):")
    print("  majority-note baseline : %.3f  (always guess '%s')"
          % (base_acc, NOTES[majority]))
    print("  n-gram model           : %.3f" % acc)
    print("  -> improvement         : +%.3f" % (acc - base_acc))
    print()
    print("Perplexity (held-out, lower = better):")
    print("  unigram baseline       : %.2f" % uni_ppl)
    print("  n-gram model           : %.2f" % ng_ppl)
    print()

    # Generate a fresh melody from a C-major seed.
    seed = [NOTE2IDX['C4'], NOTE2IDX['E4']]
    tune = model.generate(seed, length=14, rng=rng)
    print("Generated melody:", ' '.join(NOTES[n] for n in tune))
    print()

    ok = acc > base_acc + 0.2 and ng_ppl < uni_ppl * 0.6
    print("RESULT:", "PASS - n-gram beats baselines" if ok else "FAIL")
