import numpy as np

# Human Face Recognition System from scratch (Eigenfaces + nearest neighbour).
#
# Face RECOGNITION answers "WHOSE face is this?" -- a multi-class identity
# problem -- as opposed to detection ("is there a face at all?"). We synthesize
# tiny 24x24 grayscale faces for P distinct people. Each PERSON has a fixed
# latent identity: skin tone, eye spacing, eye/brow darkness, nose-bridge
# brightness, mouth height and face-oval shape. We render several photos per
# person, each corrupted by a strong DIRECTIONAL illumination ramp (a random
# light direction), plus brightness jitter, small position jitter and noise.
# Recognition must recover the identity through that nuisance.
#
# Everything is hand-rolled:
#   1) Eigenfaces = PCA on the face pixels via SVD (np.linalg allowed). The top
#      components mostly encode LIGHTING, so -- as in the classic eigenface
#      recipe -- we DROP the first few and keep the identity-bearing rest.
#   2) Project every image onto those eigenfaces and classify by
#      1-nearest-neighbour to the labelled gallery.
#
# We report held-out top-1 recognition accuracy vs (a) the majority-class
# baseline and (b) 1-NN on RAW pixels. Raw L2 distance is dominated by the
# lighting ramp and fails; the lighting-invariant eigenface space wins clearly.

P = 24                                   # image side (px)
CY, CX = 12, 12                          # nominal face centre


def blob(xx, yy, cy, cx, r):
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * r ** 2))


def make_person(rng):
    """Sample the fixed latent identity of one individual."""
    return dict(
        skin=rng.uniform(0.45, 0.88),
        eye_dx=rng.uniform(2.8, 5.8),                # eye half-spacing
        eye_y=rng.uniform(-5.2, -2.3),               # eyes above centre
        eye_dark=rng.uniform(0.18, 0.48),
        brow=rng.uniform(0.04, 0.24),
        nose=rng.uniform(0.04, 0.20),
        mouth_y=rng.uniform(3.8, 7.2),
        mouth_dark=rng.uniform(0.13, 0.36),
        ow=rng.uniform(7.2, 10.8),                   # oval width
        oh=rng.uniform(8.8, 12.2),                   # oval height
    )


def render(person, rng):
    """Render one photo of a person with lighting + jitter + noise nuisance."""
    ax = np.arange(P)
    xx, yy = np.meshgrid(ax, ax)
    jy, jx = rng.randint(-1, 2), rng.randint(-1, 2)  # small position jitter
    cy, cx = CY + jy, CX + jx
    img = 0.18 + rng.normal(0, 0.02, (P, P))         # dark background
    oval = (((xx - cx) / person["ow"]) ** 2 + ((yy - cy) / person["oh"]) ** 2) <= 1.0
    img[oval] = person["skin"]
    dx, ey = person["eye_dx"], person["eye_y"]
    img -= person["eye_dark"] * blob(xx, yy, cy + ey, cx - dx, 1.6)
    img -= person["eye_dark"] * blob(xx, yy, cy + ey, cx + dx, 1.6)
    img -= person["brow"] * blob(xx, yy, cy + ey - 2, cx - dx, 2.0)
    img -= person["brow"] * blob(xx, yy, cy + ey - 2, cx + dx, 2.0)
    img += person["nose"] * blob(xx, yy, cy, cx, 1.5)
    img -= person["mouth_dark"] * blob(xx, yy, cy + person["mouth_y"], cx, 2.2)
    # dominant nuisance: a strong directional illumination ramp
    ang = rng.uniform(0, 2 * np.pi)
    ramp = np.cos(ang) * (xx / P - 0.5) + np.sin(ang) * (yy / P - 0.5)
    img += 0.6 * ramp
    img += rng.uniform(-0.05, 0.05)                  # global brightness jitter
    img += rng.normal(0, 0.03, (P, P))               # sensor noise
    return np.clip(img, 0, 1).ravel()


class Eigenfaces:
    """PCA face space via SVD; drop the leading (lighting) components."""

    def __init__(self, k=40, n_drop=3):
        self.k, self.n_drop = k, n_drop

    def fit(self, X):
        self.mean = X.mean(0)
        A = X - self.mean                            # centre
        # SVD: rows of Vt are eigenfaces ordered by explained variance.
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        d = self.n_drop
        self.components = Vt[d:d + self.k]           # (k, d) identity eigenfaces
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T   # project onto face space


def nn_predict(gallery, labels, probes):
    """1-nearest-neighbour identity match (vectorised euclidean)."""
    d2 = ((probes[:, None, :] - gallery[None, :, :]) ** 2).sum(-1)
    return labels[d2.argmin(1)]


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    n_people, per_person, n_gallery = 16, 16, 10
    people = [make_person(rng) for _ in range(n_people)]

    X, y = [], []
    for pid, person in enumerate(people):
        for _ in range(per_person):
            X.append(render(person, rng))
            y.append(pid)
    X, y = np.array(X), np.array(y)

    # Held-out split: n_gallery labelled photos / rest as probes, per person.
    tr, te = [], []
    for pid in range(n_people):
        idx = np.where(y == pid)[0]
        rng.shuffle(idx)
        tr += list(idx[:n_gallery]); te += list(idx[n_gallery:])
    tr, te = np.array(tr), np.array(te)

    ef = Eigenfaces(k=40, n_drop=3).fit(X[tr])
    Gtr, Gte = ef.transform(X[tr]), ef.transform(X[te])

    pred_ef = nn_predict(Gtr, y[tr], Gte)            # recognition in face space
    pred_px = nn_predict(X[tr], y[tr], X[te])        # baseline: raw-pixel NN
    yte = y[te]

    acc_ef = (pred_ef == yte).mean()
    acc_px = (pred_px == yte).mean()
    counts = np.bincount(y[tr], minlength=n_people)
    majority = counts.max() / len(tr)                # guess the commonest person

    print("Synthetic faces: {} people x {} photos ({}x{} px)".format(
        n_people, per_person, P, P))
    print("Gallery {} / Probe {}   eigenfaces k={} (dropped top {})".format(
        len(tr), len(te), ef.components.shape[0], ef.n_drop))
    print("-" * 52)
    print("RECOGNITION (top-1 identity on held-out probes)")
    print("  Majority-person baseline : {:.3f}".format(majority))
    print("  Raw-pixel 1-NN           : {:.3f}".format(acc_px))
    print("  Eigenface 1-NN           : {:.3f}".format(acc_ef))
    print("  Chance level (1/{:d})       : {:.3f}".format(n_people, 1.0 / n_people))
    print("-" * 52)
    ok = acc_ef > acc_px + 0.15 and acc_ef > majority + 0.5 and acc_ef > 0.75
    print("PASS" if ok else "FAIL")
