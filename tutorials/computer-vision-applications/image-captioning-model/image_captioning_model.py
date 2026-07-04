import numpy as np

# Image captioning from scratch: an MLP encoder turns a tiny grayscale image
# into a feature vector, and a vanilla RNN decoder (trained with teacher forcing
# and manual backprop-through-time) generates a word-by-word caption describing
# the shape's SIZE, TYPE and POSITION -- three latent factors it must recover.

# Vocabulary. Captions are: START a <size> <shape> at <position> END
VOCAB = ["<S>", "<E>", "a", "at", "small", "big", "square", "disk",
         "triangle", "cross", "top", "bottom", "left", "right"]
TOK = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)

SIZES = ["small", "big"]
SHAPES = ["square", "disk", "triangle", "cross"]
POSITIONS = ["top", "bottom", "left", "right"]
CENTER = {"top": (3, 7), "bottom": (10, 7), "left": (7, 3), "right": (7, 10)}
G = 14  # canvas side


def draw(size, shape, pos):
    """Render a 14x14 grayscale image with one planted shape."""
    img = np.zeros((G, G))
    r = 2 if size == "small" else 3
    cr, cc = CENTER[pos]
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            rr, cc2 = cr + dr, cc + dc
            if not (0 <= rr < G and 0 <= cc2 < G):
                continue
            if shape == "square":
                on = True
            elif shape == "disk":
                on = dr * dr + dc * dc <= r * r
            elif shape == "triangle":                       # filled, apex up
                on = abs(dc) <= 0.6 * (dr + r)
            else:                                            # cross / plus
                on = (abs(dr) <= 1 and abs(dc) <= r) or (abs(dc) <= 1 and abs(dr) <= r)
            if on:
                img[rr, cc2] = 1.0
    return img


def make_dataset(n_per=24, noise=0.12):
    """Synthesize (image, caption) pairs over all size/shape/position combos."""
    X, C = [], []
    for size in SIZES:
        for shape in SHAPES:
            for pos in POSITIONS:
                cap = [TOK[t] for t in ("<S>", "a", size, shape, "at", pos, "<E>")]
                for _ in range(n_per):
                    img = draw(size, shape, pos)
                    img = img + noise * np.random.randn(G, G)          # additive noise
                    img *= np.random.uniform(0.85, 1.15)               # intensity jitter
                    X.append(img.ravel())
                    C.append(cap)
    return np.array(X), np.array(C)


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


class CaptioningModel:
    """MLP image encoder + vanilla-RNN caption decoder, trained end-to-end."""

    def __init__(self, D=24, Emb=16, H=48, lr=0.01, epochs=400, seed=0):
        self.D, self.Emb, self.H = D, Emb, H
        self.lr, self.epochs = lr, epochs
        rng = np.random.RandomState(seed)
        s = 0.1
        self.p = {
            "Wenc": rng.randn(G * G, D) * s, "benc": np.zeros(D),        # encoder
            "Wemb": rng.randn(V, Emb) * s,                              # word embeddings
            "Wxh": rng.randn(Emb + D, H) * s, "Whh": rng.randn(H, H) * s,
            "bh": np.zeros(H), "Why": rng.randn(H, V) * s, "by": np.zeros(V),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}       # Adam moments
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}

    def _encode(self, X):
        return np.tanh(X @ self.p["Wenc"] + self.p["benc"])             # (N, D)

    def fit(self, X, C):
        N = len(X)
        inp, tgt = C[:, :-1], C[:, 1:]                                  # teacher forcing
        T = inp.shape[1]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for ep in range(1, self.epochs + 1):
            feat = self._encode(X)
            # ---- forward, caching activations for BPTT ----
            hs, xs = [np.zeros((N, self.H))], []
            probs = []
            for t in range(T):
                emb = self.p["Wemb"][inp[:, t]]
                x = np.concatenate([emb, feat], axis=1)
                xs.append(x)
                h = np.tanh(x @ self.p["Wxh"] + hs[-1] @ self.p["Whh"] + self.p["bh"])
                hs.append(h)
                probs.append(softmax(h @ self.p["Why"] + self.p["by"]))
            # ---- backward through time ----
            g = {k: np.zeros_like(v) for k, v in self.p.items()}
            dfeat = np.zeros((N, self.D))
            dh_next = np.zeros((N, self.H))
            for t in reversed(range(T)):
                dlog = probs[t].copy()
                dlog[np.arange(N), tgt[:, t]] -= 1.0                    # softmax-CE grad
                g["Why"] += hs[t + 1].T @ dlog
                g["by"] += dlog.sum(0)
                dh = dlog @ self.p["Why"].T + dh_next
                da = dh * (1 - hs[t + 1] ** 2)                          # tanh'
                g["Wxh"] += xs[t].T @ da
                g["Whh"] += hs[t].T @ da
                g["bh"] += da.sum(0)
                dx = da @ self.p["Wxh"].T
                np.add.at(g["Wemb"], inp[:, t], dx[:, :self.Emb])
                dfeat += dx[:, self.Emb:]
                dh_next = da @ self.p["Whh"].T
            dz = dfeat * (1 - feat ** 2)                               # encoder tanh'
            g["Wenc"] += X.T @ dz
            g["benc"] += dz.sum(0)
            # ---- Adam update ----
            for k in self.p:
                gk = g[k] / N
                self.m[k] = b1 * self.m[k] + (1 - b1) * gk
                self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
                mh = self.m[k] / (1 - b1 ** ep)
                vh = self.v[k] / (1 - b2 ** ep)
                self.p[k] -= self.lr * mh / (np.sqrt(vh) + eps)
        return self

    def caption(self, X, T=6):
        """Greedy decode: feed each generated word back in (no teacher forcing)."""
        feat = self._encode(X)
        N = len(X)
        prev = np.full(N, TOK["<S>"], dtype=int)
        h = np.zeros((N, self.H))
        out = np.zeros((N, T), dtype=int)
        for t in range(T):
            x = np.concatenate([self.p["Wemb"][prev], feat], axis=1)
            h = np.tanh(x @ self.p["Wxh"] + h @ self.p["Whh"] + self.p["bh"])
            prev = np.argmax(h @ self.p["Why"] + self.p["by"], axis=1)
            out[:, t] = prev
        return out


if __name__ == "__main__":
    np.random.seed(0)

    X, C = make_dataset()
    idx = np.random.permutation(len(X))
    X, C = X[idx], C[idx]
    cut = int(0.7 * len(X))
    Xtr, Ctr, Xte, Cte = X[:cut], C[:cut], X[cut:], C[cut:]

    model = CaptioningModel().fit(Xtr, Ctr)
    pred = model.caption(Xte)
    tgt = Cte[:, 1:]                                                    # words after <S>

    # Content tokens are size (pos 1), shape (pos 2), position (pos 4).
    content = [1, 2, 4]
    tok_acc = (pred[:, content] == tgt[:, content]).mean()
    exact = np.all(pred == tgt, axis=1).mean()

    # Majority baseline: always emit the most frequent training word per slot.
    maj = np.array([np.bincount(Ctr[:, 1:][:, t]).argmax() for t in range(tgt.shape[1])])
    base_tok = (maj[content] == tgt[:, content]).mean()
    base_exact = np.all(maj == tgt, axis=1).mean()

    print("Test images:               ", len(Xte))
    print("Content-word accuracy:")
    print("  Majority baseline:        {:.3f}".format(base_tok))
    print("  Captioning model:         {:.3f}".format(tok_acc))
    print("Exact-caption match:")
    print("  Majority baseline:        {:.3f}".format(base_exact))
    print("  Captioning model:         {:.3f}".format(exact))
    print("Sample captions (model -> truth):")
    for i in range(3):
        p = " ".join(VOCAB[t] for t in pred[i])
        g = " ".join(VOCAB[t] for t in tgt[i])
        print("  [{}]  ->  [{}]".format(p, g))
    print("Beats baseline:", tok_acc > base_tok + 0.3 and exact > base_exact + 0.3)
