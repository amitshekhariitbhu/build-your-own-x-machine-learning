import numpy as np

# Document Layout Analysis: label each text block on a page as one of
# TITLE / HEADING / BODY / LIST / CAPTION / FOOTER using typographic + geometric
# cues (font size, indent, width, boldness, centering) fused with bag-of-words
# text markers ("figure", "introduction", "page", ...). Classifier is a
# softmax (multinomial logistic) regression trained from scratch by GD.

CLASSES = ["TITLE", "HEADING", "BODY", "LIST", "CAPTION", "FOOTER"]
TITLE, HEADING, BODY, LIST, CAPTION, FOOTER = range(6)

TITLE_W = ["Deep", "Learning", "Systems", "Analysis", "Networks", "Advanced",
           "Modeling", "Vision", "Language", "Methods", "Guide", "Data"]
HEAD_W = ["Introduction", "Background", "Methods", "Results", "Discussion",
          "Conclusion", "Overview", "Approach", "Experiments", "Evaluation"]
BODY_W = ["the", "model", "uses", "data", "to", "learn", "from", "examples",
          "and", "then", "predicts", "output", "for", "new", "inputs", "with",
          "accuracy", "over", "time", "using", "gradient", "updates", "each"]
LIST_W = ["configure", "dataset", "train", "model", "evaluate", "results",
          "tune", "parameters", "save", "checkpoints", "run", "script"]
CAP_W = ["shows", "results", "of", "our", "experiment", "on", "benchmark",
         "dataset", "comparison", "across", "methods"]
FOOT_W = ["confidential", "draft", "copyright"]


def tokenize(text):
    toks = []
    for t in text.lower().split():
        t = "".join(c for c in t if c.isalpha())
        if t:
            toks.append(t)
    return toks


def gen_block(rng, label):
    """Return (numeric_features, text) sampled from a class-conditioned model."""
    if label == TITLE:
        text = " ".join(rng.choice(TITLE_W, rng.randint(2, 6)))
        font, x, w = rng.normal(22, 1.5), rng.normal(0.35, .04), rng.normal(.30, .05)
        bold, cen, y = rng.rand() < .90, rng.rand() < .90, rng.normal(.05, .02)
    elif label == HEADING:
        text = " ".join(rng.choice(HEAD_W, rng.randint(1, 4)))
        if rng.rand() < .5:
            text = "%d %s" % (rng.randint(1, 9), text)      # section number
        font, x, w = rng.normal(16, 1.0), rng.normal(.08, .02), rng.normal(.40, .10)
        bold, cen, y = rng.rand() < .85, rng.rand() < .05, rng.rand()
    elif label == BODY:
        text = " ".join(rng.choice(BODY_W, rng.randint(12, 28))) + " ."
        font, x, w = rng.normal(11, .5), rng.normal(.08, .02), rng.normal(.85, .04)
        bold, cen, y = rng.rand() < .02, rng.rand() < .02, rng.rand()
    elif label == LIST:
        lead = "-" if rng.rand() < .5 else "%d )" % rng.randint(1, 9)
        text = lead + " " + " ".join(rng.choice(LIST_W, rng.randint(3, 8)))
        font, x, w = rng.normal(11, .5), rng.normal(.15, .02), rng.normal(.60, .10)
        bold, cen, y = rng.rand() < .02, rng.rand() < .02, rng.rand()
    elif label == CAPTION:
        head = "Figure" if rng.rand() < .5 else "Table"
        text = "%s %d : %s ." % (head, rng.randint(1, 9),
                                 " ".join(rng.choice(CAP_W, rng.randint(4, 10))))
        font, x, w = rng.normal(9, .5), rng.normal(.20, .05), rng.normal(.60, .10)
        bold, cen, y = rng.rand() < .05, rng.rand() < .50, rng.rand()
    else:  # FOOTER
        if rng.rand() < .7:
            text = "page %d" % rng.randint(1, 99)
        else:
            text = "%d %s" % (rng.randint(1, 99), rng.choice(FOOT_W))
        font, x, w = rng.normal(8, .5), rng.normal(.45, .05), rng.normal(.10, .03)
        bold, cen, y = rng.rand() < .02, rng.rand() < .90, rng.normal(.97, .02)

    toks = tokenize(text)
    raw = text.split()
    caps = np.mean([t[:1].isupper() for t in raw]) if raw else 0.0
    feat = [font, x, w, float(bold), float(cen), float(np.clip(y, 0, 1)),
            len(raw),                                   # n_words
            float(text.rstrip().endswith(".")),         # ends with period
            float(raw[0] == "-" if raw else 0),         # bullet lead
            float(raw[0].isdigit() if raw else 0),      # starts with number
            caps]                                       # title/caps ratio
    return np.array(feat, float), toks


def make_dataset(n_docs=30, seed=0):
    rng = np.random.RandomState(seed)
    mix = np.array([.05, .12, .45, .18, .12, .08])      # BODY is the majority class
    feats, texts, labels = [], [], []
    for _ in range(n_docs):
        for _ in range(rng.randint(16, 24)):
            lab = int(rng.choice(6, p=mix))
            f, toks = gen_block(rng, lab)
            feats.append(f); texts.append(toks); labels.append(lab)
    return np.array(feats), texts, np.array(labels)


def build_vocab(texts, min_count=2):
    counts = {}
    for toks in texts:
        for t in set(toks):
            counts[t] = counts.get(t, 0) + 1
    return {w: i for i, w in enumerate(w for w, c in counts.items() if c >= min_count)}


def featurize(feats, texts, vocab, mu, sd):
    num = (feats - mu) / sd                              # standardized geometry
    bow = np.zeros((len(texts), len(vocab)))             # binary word-presence
    for i, toks in enumerate(texts):
        for t in toks:
            if t in vocab:
                bow[i, vocab[t]] = 1.0
    return np.hstack([num, bow])


class SoftmaxLayoutClassifier:
    """Multinomial logistic regression trained with full-batch gradient descent."""

    def __init__(self, n_class, lr=0.5, n_iter=400, l2=1e-3):
        self.C, self.lr, self.n_iter, self.l2 = n_class, lr, n_iter, l2

    def _softmax(self, Z):
        Z = Z - Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    def fit(self, X, y):
        n, d = X.shape
        Y = np.eye(self.C)[y]                            # one-hot targets
        self.W = np.zeros((d, self.C))
        self.b = np.zeros(self.C)
        for _ in range(self.n_iter):
            P = self._softmax(X @ self.W + self.b)
            G = (P - Y) / n
            self.W -= self.lr * (X.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(0)
        return self

    def predict(self, X):
        return np.argmax(X @ self.W + self.b, axis=1)


def macro_f1(y, p, C):
    f1s = []
    for c in range(C):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


if __name__ == "__main__":
    np.random.seed(0)

    feats, texts, labels = make_dataset(n_docs=30, seed=0)
    n = len(labels)
    idx = np.random.permutation(n)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]

    vocab = build_vocab([texts[i] for i in tr])
    mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-8
    Xtr = featurize(feats[tr], [texts[i] for i in tr], vocab, mu, sd)
    Xte = featurize(feats[te], [texts[i] for i in te], vocab, mu, sd)
    ytr, yte = labels[tr], labels[te]

    clf = SoftmaxLayoutClassifier(n_class=6).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = float(np.mean(pred == yte))
    f1 = macro_f1(yte, pred, 6)
    maj = np.bincount(ytr, minlength=6).argmax()         # majority = BODY
    base = float(np.mean(yte == maj))

    print("Blocks: %d  (train %d / test %d)   features: %d geom + %d words"
          % (n, len(tr), len(te), feats.shape[1], len(vocab)))
    print("-" * 60)
    print("Layout classifier accuracy :  %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority baseline accuracy :  %.4f   (always '%s')"
          % (base, CLASSES[maj]))
    print("-" * 60)
    print("Per-class test recall:")
    for c in range(6):
        m = yte == c
        r = float(np.mean(pred[m] == c)) if m.any() else 0.0
        print("  %-8s n=%2d  recall=%.2f" % (CLASSES[c], int(m.sum()), r))
    print("-" * 60)
    j = int(te[0])
    print("Example block text : \"%s\"" % " ".join(texts[j]))
    print("  true=%-8s pred=%-8s" % (CLASSES[labels[j]], CLASSES[pred[0]]))
    print("Classifier beats majority baseline: %s" % (acc > base))
