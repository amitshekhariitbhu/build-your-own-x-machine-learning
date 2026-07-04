import numpy as np

# Speech Emotion Recognition from scratch: synthesize emotional "speech"
# waveforms, extract MFCC + prosodic features by hand (framing, FFT, mel
# filterbank, DCT, autocorrelation pitch), then classify with a softmax
# regression trained by gradient descent. No audio/ML libraries.

SR = 8000            # sample rate (Hz)
DUR = 0.4            # utterance length (s)
EMOTIONS = ["angry", "happy", "sad", "neutral"]

# Per-emotion prosody: base pitch F0, pitch rise over the utterance, tremolo
# rate, tremolo depth, loudness. These are the LATENT traits SER must recover.
PROFILE = {
    "angry":   dict(f0=230, rise=-10, trem=7.0, depth=0.35, energy=1.00),
    "happy":   dict(f0=250, rise=60,  trem=5.0, depth=0.20, energy=0.80),
    "sad":     dict(f0=130, rise=-25, trem=2.0, depth=0.10, energy=0.35),
    "neutral": dict(f0=175, rise=5,   trem=3.0, depth=0.08, energy=0.60),
}


def synth_utterance(emotion, rng):
    """Build one harmonic voice signal whose prosody encodes the emotion."""
    p = PROFILE[emotion]
    n = int(SR * DUR)
    t = np.arange(n) / SR
    # Time-varying pitch (+ speaker jitter) and a tremolo loudness envelope.
    f0 = p["f0"] + p["rise"] * (t / DUR) + rng.normal(0, 6)
    trem = 1.0 + p["depth"] * np.sin(2 * np.pi * (p["trem"] + rng.normal(0, 0.4)) * t)
    env = np.hanning(n) * trem * (p["energy"] * (1 + rng.normal(0, 0.08)))
    # Sum of harmonics -> a voiced, vowel-like timbre.
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(np.sin(k * phase) / k for k in range(1, 6))
    sig = env * sig + rng.normal(0, 0.02, n)      # + breath noise
    return sig


def mel_filterbank(n_fft, n_mels=20):
    """Triangular filters spaced evenly on the mel scale over [0, SR/2]."""
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700.0)
    mel2hz = lambda m: 700 * (10 ** (m / 2595.0) - 1)
    edges = mel2hz(np.linspace(hz2mel(0), hz2mel(SR / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, mid):
            fb[m - 1, k] = (k - lo) / max(mid - lo, 1)
        for k in range(mid, hi):
            fb[m - 1, k] = (hi - k) / max(hi - mid, 1)
    return fb


def dct_matrix(n_out, n_in):
    """DCT-II basis used to decorrelate log-mel energies into cepstral coeffs."""
    k = np.arange(n_out)[:, None]
    j = np.arange(n_in)[None, :]
    return np.cos(np.pi * k * (2 * j + 1) / (2 * n_in))


FRAME, HOP, NMEL, NCEP = 256, 128, 20, 13
_FB = mel_filterbank(FRAME, NMEL)
_DCT = dct_matrix(NCEP, NMEL)
_WIN = np.hamming(FRAME)


def features(sig):
    """MFCC mean/std over frames + prosodic pitch, energy, zero-cross rate."""
    frames = [sig[i:i + FRAME] * _WIN
              for i in range(0, len(sig) - FRAME, HOP)]
    frames = np.array(frames)
    mag = np.abs(np.fft.rfft(frames, axis=1))          # magnitude spectrum
    mel = np.log((mag ** 2) @ _FB.T + 1e-8)            # log mel-band energy
    mfcc = mel @ _DCT.T                                 # (n_frames, NCEP)
    feat = np.concatenate([mfcc.mean(0), mfcc.std(0)])
    # Prosody: autocorrelation pitch, mean energy, zero-crossing rate.
    r = np.correlate(sig, sig, "full")[len(sig) - 1:]
    lo, hi = SR // 400, SR // 60                        # 60..400 Hz search band
    pitch = SR / (lo + np.argmax(r[lo:hi]))
    energy = np.sqrt(np.mean(sig ** 2))
    zcr = np.mean(np.abs(np.diff(np.sign(sig)))) / 2
    return np.concatenate([feat, [pitch, energy, zcr]])


def make_dataset(n_per=70, seed=0):
    rng = np.random.RandomState(seed)
    X, y = [], []
    for label, emo in enumerate(EMOTIONS):
        for _ in range(n_per):
            X.append(features(synth_utterance(emo, rng)))
            y.append(label)
    return np.array(X), np.array(y)


class SoftmaxSER:
    """Multinomial logistic regression (softmax) via full-batch gradient descent."""

    def __init__(self, lr=0.3, epochs=400, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xb = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        C = int(y.max()) + 1
        Y = np.eye(C)[y]
        self.W = np.zeros((Xb.shape[1], C))
        for _ in range(self.epochs):
            z = Xb @ self.W
            z -= z.max(1, keepdims=True)
            P = np.exp(z); P /= P.sum(1, keepdims=True)
            grad = Xb.T @ (P - Y) / len(Xb) + self.l2 * self.W
            self.W -= self.lr * grad
        return self

    def predict(self, X):
        Xb = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        return (Xb @ self.W).argmax(1)


def macro_f1(y, p, C):
    fs = []
    for c in range(C):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return np.mean(fs)


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_dataset(n_per=70)
    idx = np.random.permutation(len(X))
    split = int(0.7 * len(X))
    tr, te = idx[:split], idx[split:]

    clf = SoftmaxSER().fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    acc = np.mean(pred == y[te])
    f1 = macro_f1(y[te], pred, len(EMOTIONS))

    # Majority-class baseline: always predict the most common training emotion.
    maj = np.bincount(y[tr]).argmax()
    base_pred = np.full_like(y[te], maj)
    base_acc = np.mean(base_pred == y[te])
    base_f1 = macro_f1(y[te], base_pred, len(EMOTIONS))

    print("Utterances: %d   Train: %d   Test: %d   Classes: %d"
          % (len(X), len(tr), len(te), len(EMOTIONS)))
    print("Features/utterance: %d (MFCC mean+std + pitch,energy,zcr)" % X.shape[1])
    print("-" * 60)
    print("Softmax SER    accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("Random guess   accuracy: %.4f" % (1.0 / len(EMOTIONS)))
    print("-" * 60)
    print("Per-emotion recall on test set:")
    for c, emo in enumerate(EMOTIONS):
        m = y[te] == c
        rec = np.mean(pred[m] == c) if m.any() else 0.0
        print("  %-8s recall: %.2f" % (emo, rec))
    print("-" * 60)
    print("Softmax SER beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
