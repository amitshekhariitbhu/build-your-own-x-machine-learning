import numpy as np

# Build a Video Summarization System from scratch.
# A video is a sequence of frames; each frame is reduced to a compact
# descriptor (a normalized colour histogram). We plant S distinct SCENES of
# UNEQUAL length: inside a scene every frame shares one content signature
# (plus mild noise and slow drift); at each cut the signature changes
# abruptly. Task: pick K keyframes that best REPRESENT the whole clip.
# We cluster the frame descriptors with k-means (from scratch, k-means++
# seeding) and keep the MEDOID frame of each cluster as a keyframe. Because
# clustering follows CONTENT rather than the clock, it grabs one frame from
# every scene -- even the short ones -- while uniform/random sampling
# over-covers long scenes and drops short ones. We score with (1) SCENE
# COVERAGE and (2) mean REPRESENTATION error (frame -> nearest keyframe),
# both against the two time-based baselines.

D = 16                 # histogram bins per frame descriptor
S = 6                  # planted scenes
LENGTHS = [30, 6, 22, 5, 34, 8]     # UNEQUAL scene lengths (105 frames total)


def make_video(seed=0):
    # Synthetic clip: each scene owns a distinct histogram signature.
    rng = np.random.RandomState(seed)
    sigs = np.full((S, D), 0.2)
    for s in range(S):                       # give every scene its own peaks
        peak = (s * 3) % D
        sigs[s, peak] += 5.0
        sigs[s, (peak + 1) % D] += 2.0
    sigs /= sigs.sum(axis=1, keepdims=True)

    frames, labels = [], []
    for s, L in enumerate(LENGTHS):
        drift = np.linspace(0, 1, L)[:, None] * rng.randn(1, D) * 0.01
        block = sigs[s][None, :] + rng.randn(L, D) * 0.015 + drift
        block = np.clip(block, 1e-6, None)
        block /= block.sum(axis=1, keepdims=True)   # keep valid histograms
        frames.append(block)
        labels += [s] * L
    return np.vstack(frames), np.array(labels)


def sqdist(A, B):
    # Pairwise squared Euclidean distances between rows of A and rows of B.
    D2 = (np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :]
          - 2.0 * A @ B.T)
    return np.maximum(D2, 0.0)


class VideoSummarizer:
    """Content-based keyframe selector: k-means clusters + medoid frames."""

    def __init__(self, n_keyframes=S, n_iter=50, seed=0):
        self.k = n_keyframes
        self.n_iter = n_iter
        self.seed = seed

    def _kmeanspp(self, X, rng):
        # k-means++ seeding: spread initial centers by squared-distance.
        idx = [rng.randint(len(X))]
        d2 = sqdist(X, X[idx[0]][None])[:, 0]
        for _ in range(1, self.k):
            nxt = rng.choice(len(X), p=d2 / d2.sum())
            idx.append(nxt)
            d2 = np.minimum(d2, sqdist(X, X[nxt][None])[:, 0])
        return X[idx].copy()

    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        C = self._kmeanspp(X, rng)
        for _ in range(self.n_iter):
            assign = sqdist(X, C).argmin(1)
            newC = np.array([X[assign == j].mean(0) if np.any(assign == j)
                             else C[j] for j in range(self.k)])
            if np.allclose(newC, C):
                break
            C = newC
        # Keyframe = the real frame nearest its cluster center (medoid).
        keys = []
        for j in range(self.k):
            m = np.nonzero(assign == j)[0]
            keys.append(m[sqdist(X[m], C[j][None])[:, 0].argmin()])
        self.keyframes = np.sort(np.array(keys))
        return self


def coverage(keys, labels):
    # Fraction of planted scenes represented by at least one keyframe.
    return len(np.unique(labels[keys])) / S


def rep_error(keys, X):
    # Mean distance from each frame to its nearest selected keyframe.
    return np.sqrt(sqdist(X, X[keys]).min(1).clip(min=0)).mean()


if __name__ == "__main__":
    np.random.seed(0)

    X, labels = make_video(seed=0)
    T = len(X)

    keys = VideoSummarizer(n_keyframes=S).fit(X).keyframes
    sm_cov, sm_err = coverage(keys, labels), rep_error(keys, X)

    # Baseline 1: uniform sampling -- keyframes evenly spaced in TIME.
    uni = np.unique(np.linspace(0, T - 1, S).round().astype(int))
    uni_cov, uni_err = coverage(uni, labels), rep_error(uni, X)

    # Baseline 2: random sampling, averaged over many trials.
    rng = np.random.RandomState(1)
    r_cov, r_err = [], []
    for _ in range(200):
        r = rng.choice(T, S, replace=False)
        r_cov.append(coverage(r, labels))
        r_err.append(rep_error(r, X))
    r_cov, r_err = np.mean(r_cov), np.mean(r_err)

    print("Frames: %d   Planted scenes: %d (lengths %s)   Keyframes: %d"
          % (T, S, LENGTHS, S))
    print("-" * 62)
    print("Method              scene-coverage   representation-error")
    print("Summarizer (kmeans)     %.3f              %.4f" % (sm_cov, sm_err))
    print("Baseline (uniform)      %.3f              %.4f" % (uni_cov, uni_err))
    print("Baseline (random avg)   %.3f              %.4f" % (r_cov, r_err))
    print("-" * 62)
    print("Selected keyframes (frame idx -> scene):",
          list(zip(keys.tolist(), labels[keys].tolist())))
    assert sm_cov == 1.0, "summary failed to cover every scene"
    assert sm_cov > uni_cov and sm_cov > r_cov, "did not beat coverage baselines"
    assert sm_err < uni_err and sm_err < r_err, "did not beat error baselines"
    print("PASS: summary covers ALL scenes and beats both time-based baselines.")
