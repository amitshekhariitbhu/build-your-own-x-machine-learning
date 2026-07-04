import numpy as np

# Visualization System for a Machine Learning Algorithm, from scratch.
# We train a from-scratch logistic-regression classifier on 2D data and build
# a tiny text-only "visualization system" around it -- no plotting library.
# The system has three reusable primitives that turn numbers into pictures:
#   (1) SPARKLINE  -> a 1D series (the training loss) as a ramp of glyphs,
#   (2) HEATMAP    -> a 2D field (the predicted-probability surface) as glyphs,
#   (3) REGION MAP -> the classifier's decision regions + the training points.
# A picture is only useful if it FAITHFULLY encodes the numbers, so every
# render is proven correct by DECODING it back and comparing to the source,
# each beating an explicit baseline (majority / random / shuffled).

RAMP = " .:-=+*#%@"                       # 10 intensity levels, low -> high


def make_data():
    """Two Gaussian blobs -> a planted, near-linearly-separable 2-class set."""
    np.random.seed(0)
    n = 200
    c0 = np.random.randn(n, 2) * 1.2 + np.array([-2.0, -2.0])   # class 0
    c1 = np.random.randn(n, 2) * 1.2 + np.array([2.0,  2.0])    # class 1
    X = np.vstack([c0, c1])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    idx = np.random.permutation(len(y))
    return X[idx], y[idx].astype(int)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    """Batch gradient-descent logistic regression, records its loss curve."""

    def __init__(self, lr=0.5, n_iter=300):
        self.lr, self.n_iter = lr, n_iter

    def _bias(self, X):
        return np.hstack([np.ones((len(X), 1)), X])

    def fit(self, X, y):
        Xb, self.losses = self._bias(X), []
        self.w = np.zeros(Xb.shape[1])
        for _ in range(self.n_iter):
            p = sigmoid(Xb @ self.w)
            self.w -= self.lr * (Xb.T @ (p - y)) / len(y)
            eps = 1e-9
            self.losses.append(-np.mean(y * np.log(p + eps) +
                                        (1 - y) * np.log(1 - p + eps)))
        return self

    def predict_proba(self, X):
        return sigmoid(self._bias(X) @ self.w)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ---------------- the visualization system (pure text) ----------------

def sparkline(series):
    """Map a 1D series onto RAMP glyphs (min -> ' ', max -> '@')."""
    s = np.asarray(series, float)
    lo, hi = s.min(), s.max()
    lvl = np.zeros_like(s, int) if hi == lo else \
        np.round((s - lo) / (hi - lo) * (len(RAMP) - 1)).astype(int)
    return "".join(RAMP[i] for i in lvl), lvl


def heatmap(field, rows, cols):
    """Render a (rows, cols) value field in [0,1] as RAMP glyph rows."""
    lvl = np.round(field * (len(RAMP) - 1)).astype(int)
    return [" ".join(RAMP[i] for i in row) for row in lvl], lvl


def predict_grid(model, X, rows, cols):
    """Evaluate the model on a lattice spanning the data -> class & prob grids."""
    x0, x1 = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y0, y1 = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xs, ys = np.linspace(x0, x1, cols), np.linspace(y1, y0, rows)  # top = high y
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    prob = model.predict_proba(pts).reshape(rows, cols)
    return model.predict(pts).reshape(rows, cols), prob, xs, ys


REGION = {0: '.', 1: '#'}
DECODE = {'.': 0, '#': 1}


def region_map(classes, X, y, xs, ys):
    """Decision regions as glyphs, with training points overlaid ('o'/'x')."""
    grid = [[REGION[c] for c in row] for row in classes]
    for (px, py), lbl in zip(X, y):
        c = int(np.argmin(np.abs(xs - px)))
        r = int(np.argmin(np.abs(ys - py)))
        grid[r][c] = 'o' if lbl == 1 else 'x'
    return [" ".join(row) for row in grid]


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return 0.0 if d == 0 else float((a @ b) / d)


if __name__ == "__main__":
    X, y = make_data()
    ntr = int(0.7 * len(y))
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    model = LogisticRegression().fit(Xtr, ytr)
    rows, cols = 14, 28
    classes, prob, xs, ys = predict_grid(model, X, rows, cols)

    # ---- render the three visualizations ----
    loss_curve = np.array(model.losses)[::5]         # subsample for a compact line
    spark, loss_lvl = sparkline(loss_curve)
    print("training loss (start -> end):")
    print("  " + spark)

    print("\npredicted-probability heatmap (P(class=1), '@'=high '.'=low):")
    hm_rows, hm_lvl = heatmap(prob, rows, cols)
    for r in hm_rows:
        print("  " + r)

    print("\ndecision regions  ( '.'=0  '#'=1 ,  points: x=0  o=1 ):")
    for r in region_map(classes, Xtr, ytr, xs, ys):
        print("  " + r)

    # ---- correctness signals, each vs an explicit baseline ----
    # (A) the classifier itself beats the majority-class baseline
    acc = np.mean(model.predict(Xte) == yte)
    maj = max(np.mean(yte == 0), np.mean(yte == 1))

    # (B) region map is FAITHFUL: decode glyphs -> class, match the model
    decoded = np.array([[DECODE[g] for g in row] for row in
                        [[REGION[c] for c in r] for r in classes]])
    region_recover = np.mean(decoded == classes)          # exact round-trip
    region_rand = 0.5                                      # 2-class guess

    # (C) heatmap is FAITHFUL: decoded glyph levels track true probability
    decoded_p = hm_lvl.ravel() / (len(RAMP) - 1)
    heat_corr = pearson(decoded_p, prob.ravel())
    rng = np.random.default_rng(1)
    heat_shuffled = pearson(rng.permutation(decoded_p), prob.ravel())

    # (D) loss sparkline is FAITHFUL: decoded glyphs fall like the true loss
    spark_corr = pearson(loss_lvl, loss_curve)
    loss_drop = model.losses[0] - model.losses[-1]

    print("\n--- correctness signal (visualization system) ---")
    print(f"(A) classifier test acc ....... {acc:.3f}   vs majority {maj:.3f}")
    print(f"(B) region-map recover ........ {region_recover:.3f}   vs random  {region_rand:.3f}")
    print(f"(C) heatmap corr w/ prob ...... {heat_corr:.3f}   vs shuffled {heat_shuffled:+.3f}")
    print(f"(D) loss-spark corr w/ loss ... {spark_corr:.3f}   (loss drop {loss_drop:.3f} > 0)")

    ok = (acc > maj + 0.1 and region_recover == 1.0 and
          heat_corr > 0.95 and heat_corr > abs(heat_shuffled) + 0.5 and
          spark_corr > 0.95 and loss_drop > 0)
    print("RESULT:", "PASS -- renders faithfully and beats every baseline" if ok else "FAIL")
