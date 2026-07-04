import numpy as np

# Text-to-Speech (TTS) from scratch: text -> phonemes (a grapheme-to-phoneme
# lexicon) -> a source-filter SYNTHESIZER that turns each phoneme into audio
# (formant resonators for vowels, band-passed noise for fricatives) -> a
# concatenated speech waveform. No audio/ML libraries.
#
# We can't "listen" in a script, so we PROVE the speech is intelligible by
# decoding it back: a phoneme classifier trained only on our own synthesizer
# recovers the intended phonemes/words from the raw audio. High recognition
# accuracy vs a random baseline shows the waveform faithfully encodes the text.

SR = 8000                 # sample rate (Hz)
SEG = 0.12                # seconds of audio per phoneme
NSEG = int(SR * SEG)      # samples per phoneme

# ---- Phoneme inventory: the LATENT structure the decoder must recover ----
# Voiced phonemes = vocal-tract formants (F1,F2,F3) + a source spectral tilt.
# Unvoiced fricatives = a noise passband (low_hz, high_hz).
VOWEL = {  # (F1, F2, F3, tilt)
    "AA": (700, 1220, 2600, 1.0), "IY": (280, 2250, 2890, 1.0),
    "UW": (310, 870, 2250, 1.0),  "EH": (530, 1840, 2480, 1.0),
    "AO": (570, 840, 2410, 1.0),  "ER": (490, 1350, 1690, 1.0),
    "M":  (280, 1150, 2400, 2.2), "L":  (360, 1300, 2900, 1.3),
}
FRIC = {"S": (2900, 4000), "SH": (1600, 2800), "F": (400, 1500)}
PHONES = list(VOWEL) + list(FRIC)
P2I = {p: i for i, p in enumerate(PHONES)}
K = len(PHONES)

# ---- Lexicon = grapheme-to-phoneme dictionary (synthetic words) ----
LEXICON = {
    "SAAM": ["S", "AA", "M"], "LIYF": ["L", "IY", "F"], "SHUW": ["SH", "UW"],
    "FEHL": ["F", "EH", "L"], "MAOR": ["M", "AO", "ER"], "SIYS": ["S", "IY", "S"],
    "LAAF": ["L", "AA", "F"], "SHEHM": ["SH", "EH", "M"],
}


def reson(f, cf, bw=100.0):
    """Formant resonance gain (Lorentzian peak at center freq cf)."""
    return 1.0 / (1.0 + ((f - cf) / bw) ** 2)


def synth_phoneme(ph, rng):
    """Source-filter synthesis of one phoneme into a waveform segment."""
    t = np.arange(NSEG) / SR
    if ph in VOWEL:
        f1, f2, f3, tilt = VOWEL[ph]
        f0 = 120 + rng.normal(0, 4)                  # glottal pitch + jitter
        sig = np.zeros(NSEG)
        for k in range(1, int((SR / 2) / f0) + 1):   # sum harmonics of F0
            fk = k * f0
            g = reson(fk, f1) + reson(fk, f2) + 0.6 * reson(fk, f3)
            g /= k ** tilt                            # source spectral tilt
            sig += g * np.sin(2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi))
    else:                                             # fricative = colored noise
        lo, hi = FRIC[ph]
        X = np.fft.rfft(rng.normal(0, 1, NSEG))
        fr = np.fft.rfftfreq(NSEG, 1 / SR)
        X[(fr < lo) | (fr > hi)] = 0                  # band-pass shapes the hiss
        sig = np.fft.irfft(X, NSEG)
    sig *= np.hanning(NSEG)                            # taper for clean concat
    sig += rng.normal(0, 0.01, NSEG)                  # breath noise
    return sig / (np.sqrt(np.mean(sig ** 2)) + 1e-9)  # RMS normalize


def synthesize(word, rng):
    """text -> phoneme sequence (G2P) -> concatenated speech waveform."""
    return np.concatenate([synth_phoneme(p, rng) for p in LEXICON[word]])


# ---- Acoustic features: log mel-band energies of a segment's spectrum ----
def mel_fb(n_fft, n_mels=26):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700.0)
    mel2hz = lambda m: 700 * (10 ** (m / 2595.0) - 1)
    e = mel2hz(np.linspace(hz2mel(0), hz2mel(SR / 2), n_mels + 2))
    b = np.floor((n_fft + 1) * e / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        for k in range(b[m - 1], b[m]):
            fb[m - 1, k] = (k - b[m - 1]) / max(b[m] - b[m - 1], 1)
        for k in range(b[m], b[m + 1]):
            fb[m - 1, k] = (b[m + 1] - k) / max(b[m + 1] - b[m], 1)
    return fb


NFFT = 1024
MEL = mel_fb(NFFT)


def feats(seg):
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), NFFT)) ** 2
    return np.log(MEL @ S + 1e-8)


class Softmax:
    """Multinomial logistic regression via full-batch gradient descent."""

    def fit(self, X, y, iters=300, lr=0.5, l2=1e-3):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        Y = np.eye(K)[y]
        self.W = np.zeros((Xs.shape[1], K)); self.b = np.zeros(K)
        for _ in range(iters):
            Z = Xs @ self.W + self.b
            P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            G = (P - Y) / len(Xs)
            self.W -= lr * (Xs.T @ G + l2 * self.W); self.b -= lr * G.sum(0)
        return self

    def predict(self, X):
        return np.argmax(((X - self.mu) / self.sd) @ self.W + self.b, 1)


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    # ---- Train the phoneme decoder on isolated synthesized phonemes ----
    Xtr, ytr = [], []
    for p in PHONES:
        for _ in range(60):
            Xtr.append(feats(synth_phoneme(p, rng))); ytr.append(P2I[p])
    clf = Softmax().fit(np.array(Xtr), np.array(ytr))

    # ---- Held-out test: synthesize whole WORDS, then decode them back ----
    words, reps = list(LEXICON), 20
    ph_ok = ph_tot = w_ok = w_tot = 0
    counts = np.zeros(K)
    for _ in range(reps):
        for w in words:
            seq = LEXICON[w]
            wav = synthesize(w, rng)                  # <-- the TTS output audio
            preds = [PHONES[clf.predict(feats(s)[None])[0]]
                     for s in np.split(wav, len(seq))]  # uniform segmentation
            ph_ok += sum(a == b for a, b in zip(preds, seq)); ph_tot += len(seq)
            for p in seq:
                counts[P2I[p]] += 1
            w_ok += (preds == seq); w_tot += 1

    ph_acc = ph_ok / ph_tot
    w_acc = w_ok / w_tot
    maj = counts.max() / counts.sum()                 # majority-phoneme baseline

    # ---- Show the TTS pipeline on one example ----
    demo = "LIYF"
    wav = synthesize(demo, rng)
    print("=== From-scratch Text-to-Speech ===")
    print(f"text '{demo}'  ->  phonemes {LEXICON[demo]}  ->  "
          f"{len(wav)} samples ({len(wav)/SR:.2f}s of audio)")
    print(f"vocabulary: {K} phonemes, {len(words)} words\n")
    print("--- Intelligibility (decode the synthesized audio back) ---")
    print(f"Phoneme accuracy: {ph_acc:6.1%}  "
          f"(random {1/K:.1%}, majority {maj:.1%})")
    print(f"Word accuracy:    {w_acc:6.1%}  (random {1/len(words):.1%})")
    print("Result:", "PASS -- synthesized speech is intelligible"
          if ph_acc > 3 * maj and w_acc > 3 * (1 / len(words)) else "FAIL")
