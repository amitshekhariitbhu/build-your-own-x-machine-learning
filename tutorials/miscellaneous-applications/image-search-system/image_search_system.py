import numpy as np

# Build an Image Search System (content-based image retrieval) from scratch.
# We synthesize a labelled image bank: each image belongs to one visual CLASS
# defined by a dominant COLOR tint and a striped/patterned TEXTURE (horizontal,
# vertical, diagonal, anti-diagonal, checker, radial). Every image also gets its
# own random brightness, phase shift and pixel noise, so within-class images look
# related but never identical. For each image we hand-compute a descriptor:
#   (1) a per-channel COLOR histogram   -> captures the dominant tint, and
#   (2) a gradient-ORIENTATION histogram (a tiny global HOG) -> captures the
#       texture direction.
# The descriptor is L2-normalized so a dot product equals cosine similarity.
# Search = rank the whole bank by similarity to a query. We evaluate on a
# HELD-OUT set of query images (never indexed) with mean Average Precision and
# Precision@k, where "relevant" means same class, and beat a random-ranking
# baseline (whose expected score is just the class prevalence).

H, W = 24, 24                       # image height / width
CBINS, OBINS = 8, 9                 # color-histogram bins, orientation bins
PATTERNS = ["horiz", "vert", "diag", "anti", "checker", "radial"]
COLORS = np.array([[0.90, 0.25, 0.25],   # red      class palette (one per class)
                   [0.25, 0.80, 0.35],   # green
                   [0.30, 0.45, 0.95],   # blue
                   [0.95, 0.85, 0.20],   # yellow
                   [0.75, 0.30, 0.85],   # purple
                   [0.20, 0.80, 0.85]])  # cyan


def _pattern(kind, phase):
    # Grayscale intensity map in [0,1] with a class-specific stripe direction.
    y, x = np.mgrid[0:H, 0:W] * (2 * np.pi / 8.0)
    if kind == "horiz":    g = np.sin(y + phase)
    elif kind == "vert":   g = np.sin(x + phase)
    elif kind == "diag":   g = np.sin(x + y + phase)
    elif kind == "anti":   g = np.sin(x - y + phase)
    elif kind == "checker":g = np.sin(x + phase) * np.sin(y + phase)
    else:                  g = np.sin(np.hypot(x - W * np.pi / 8, y - H * np.pi / 8) + phase)
    return 0.5 + 0.5 * g


def make_image(cls, rng):
    # Compose a class image: colored texture + brightness jitter + noise.
    inten = _pattern(PATTERNS[cls], rng.uniform(0, 2 * np.pi))
    bright = rng.uniform(0.8, 1.15)
    img = COLORS[cls] * (0.35 + 0.65 * inten)[..., None] * bright
    img = img + rng.randn(H, W, 3) * 0.05
    return np.clip(img, 0.0, 1.0)


def make_bank(n_per_class, rng):
    # Generate a labelled set of images, one planted class after another.
    imgs, labels = [], []
    for cls in range(len(PATTERNS)):
        for _ in range(n_per_class):
            imgs.append(make_image(cls, rng))
            labels.append(cls)
    return np.array(imgs), np.array(labels)


def descriptor(img):
    # Hand-built content descriptor: color histogram + gradient-orientation HOG.
    feat = []
    for c in range(3):                                   # color: 3 x CBINS bins
        h, _ = np.histogram(img[:, :, c], bins=CBINS, range=(0, 1))
        feat.append(h)
    gray = img.mean(axis=2)                              # texture: global HOG
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    ori = (np.arctan2(gy, gx) % np.pi)                   # undirected: [0, pi)
    idx = np.minimum((ori / np.pi * OBINS).astype(int), OBINS - 1)
    hog = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=OBINS)
    feat.append(hog)
    v = np.concatenate(feat).astype(float)
    return v / (np.linalg.norm(v) or 1.0)                # L2-normalize -> cosine


class ImageSearch:
    """Content-based image retrieval: index descriptors, rank by cosine sim."""

    def fit(self, imgs, labels):
        self.labels = np.asarray(labels)
        self.index = np.array([descriptor(im) for im in imgs])   # (N, D)
        return self

    def query(self, img, k=None):
        # Return database indices ranked by descending similarity to `img`.
        sims = self.index @ descriptor(img)
        order = np.argsort(-sims)
        return order if k is None else order[:k]


def average_precision(ranked_labels, cls):
    # AP for one query: relevant = same class, over the full ranking.
    rel = (ranked_labels == cls).astype(float)
    if rel.sum() == 0:
        return 0.0
    prec_at = np.cumsum(rel) / (np.arange(len(rel)) + 1)
    return (prec_at * rel).sum() / rel.sum()


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    # Indexed image bank + a HELD-OUT query set (same classes, unseen images).
    train_imgs, train_lab = make_bank(n_per_class=22, rng=rng)
    query_imgs, query_lab = make_bank(n_per_class=6, rng=rng)

    engine = ImageSearch().fit(train_imgs, train_lab)

    K = 5
    ap, p_at_k = [], []
    for qi, qimg in enumerate(query_imgs):
        order = engine.query(qimg)
        ranked = engine.labels[order]
        ap.append(average_precision(ranked, query_lab[qi]))
        p_at_k.append(np.mean(ranked[:K] == query_lab[qi]))
    mAP = float(np.mean(ap))
    p_at_k = float(np.mean(p_at_k))
    top1 = float(np.mean([engine.labels[engine.query(q, 1)[0]] == query_lab[i]
                          for i, q in enumerate(query_imgs)]))

    # Random-ranking baseline: expected precision == class prevalence.
    prevalence = np.mean(train_lab == query_lab[0])      # balanced classes
    rng2 = np.random.RandomState(1)
    rand_ap = np.mean([average_precision(engine.labels[rng2.permutation(len(engine.labels))],
                                         query_lab[i]) for i in range(len(query_imgs))])

    print("Indexed images: %d   Queries (held-out): %d   Classes: %d   Descriptor dim: %d"
          % (len(train_lab), len(query_lab), len(PATTERNS), engine.index.shape[1]))
    print("-" * 62)
    print("Image search   mAP: %.3f   P@%d: %.3f   Top-1 acc: %.3f"
          % (mAP, K, p_at_k, top1))
    print("Random rank    mAP: %.3f   P@%d: %.3f   Top-1 acc: %.3f"
          % (rand_ap, K, prevalence, prevalence))
    print("mAP improvement over random: %.1fx" % (mAP / prevalence))
    print("-" * 62)
    print("SUCCESS" if (mAP > 0.8 and p_at_k > 0.8 and mAP > 3 * prevalence) else "FAIL")
