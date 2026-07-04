import numpy as np

# Code syntax highlighting as token classification: every token in a code
# snippet gets a syntactic category (keyword / function / variable / number /
# string / operator / punctuation / comment).  A from-scratch multinomial
# logistic regression (softmax + gradient descent) scores hand-crafted context
# features, and snippets are decoded greedily left-to-right so the previous
# *predicted* label feeds the next token (needed to carry comments forward).
#
# The planted twist that forces real context use, not token lookup:
#   * identifier names are SHARED between function-call and variable roles, so
#     "data" is FUNC in `data ( x )` but VAR in `data = 1` -- only the next
#     token "(" tells them apart;
#   * comment words are drawn from the SAME pool as code words, so "total" is
#     COM inside `# total score` but VAR in code -- only the carried prev-label
#     "COM" tells them apart.

LABELS = ["KW", "FUNC", "VAR", "NUM", "STR", "OP", "PUN", "COM"]
LIDX = {l: i for i, l in enumerate(LABELS)}

OPS1 = list("=+-*/<>%")
OPS2 = ["==", "!=", "<=", ">=", "+=", "-=", "//", "**"]
OPSET = set(OPS1) | set(OPS2)
PUNSET = set("()[]{}:,.")


def feats(tokens, i, prev_lab):
    # Features fired for token i; "next" carries the FUNC-vs-VAR signal and
    # "prevlab" carries the sequence/comment signal.  Shape flags let the model
    # generalise to never-seen numbers and strings.
    t = tokens[i]
    tl = t.lower()
    return ["bias",
            "tok=" + tl,
            "pre2=" + tl[:2],
            "suf2=" + tl[-2:],
            "isdigit=" + str(t.replace(".", "", 1).isdigit()),
            "isquote=" + str(t[:1] == '"'),
            "isalpha=" + str(t.isalpha()),
            "ispunct=" + str(t in PUNSET),
            "isop=" + str(t in OPSET),
            "next=" + (tokens[i + 1] if i + 1 < len(tokens) else "</s>"),
            "prev=" + (tokens[i - 1].lower() if i > 0 else "<s>"),
            "prevlab=" + prev_lab]


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


class SyntaxHighlighter:
    """Softmax token tagger trained by full-batch gradient descent."""

    def __init__(self, iters=250, lr=0.5, reg=1e-4):
        self.iters, self.lr, self.reg = iters, lr, reg

    def fit(self, data):
        # Index every feature seen in training (with GOLD previous label).
        self.fidx, rows, ys = {}, [], []
        for tokens, labs in data:
            for i in range(len(tokens)):
                pl = labs[i - 1] if i > 0 else "<start>"
                idxs = []
                for f in feats(tokens, i, pl):
                    j = self.fidx.setdefault(f, len(self.fidx))
                    idxs.append(j)
                rows.append(idxs)
                ys.append(LIDX[labs[i]])

        N, D, C = len(rows), len(self.fidx), len(LABELS)
        X = np.zeros((N, D))
        for r, idxs in enumerate(rows):
            X[r, idxs] = 1.0
        Y = np.eye(C)[np.array(ys)]

        # Cross-entropy minimisation: W -= lr * (X^T (P - Y)/N + reg*W).
        self.W = np.zeros((D, C))
        for _ in range(self.iters):
            P = softmax(X @ self.W)
            grad = X.T @ (P - Y) / N + self.reg * self.W
            self.W -= self.lr * grad
        return self

    def _score(self, tokens, i, prev_lab):
        s = np.zeros(len(LABELS))
        for f in feats(tokens, i, prev_lab):
            j = self.fidx.get(f)
            if j is not None:
                s += self.W[j]
        return s

    def predict(self, tokens):
        # Greedy left-to-right: predicted prev label feeds the next token.
        prev, out = "<start>", []
        for i in range(len(tokens)):
            lab = LABELS[int(np.argmax(self._score(tokens, i, prev)))]
            out.append(lab)
            prev = lab
        return out


# --------------------------- synthetic code corpus ---------------------------

IDENTS = ["data", "result", "count", "items", "value", "total", "index",
          "node", "buffer", "user", "score", "queue", "cache", "state"]
BUILTINS = ["print", "len", "range", "sum", "min", "max", "sorted", "open"]
FUNCS = IDENTS + BUILTINS            # names shared with variables -> ambiguous
KEYWORDS = ["if", "for", "while", "return", "in", "and", "or", "not"]
CWORDS = ["compute", "the", "loop", "over", "check", "fix"] + IDENTS  # comments


def make_corpus(n=280, seed=0):
    # Each sample is one statement: a list of (token, label) pairs.  Structure
    # is planted so labels are recoverable from context, not the token alone.
    rng = np.random.RandomState(seed)

    def num():
        return ("%d" % rng.randint(0, 100)) if rng.rand() < 0.7 \
            else "%.2f" % (rng.rand() * 10)

    def string():
        return '"' + str(rng.choice(CWORDS)) + '"'

    def atom():
        r = rng.rand()
        if r < 0.55:
            return [(str(rng.choice(IDENTS)), "VAR")]
        if r < 0.8:
            return [(num(), "NUM")]
        return [(string(), "STR")]

    def call():                       # FUNC ( arg , arg )
        toks = [(str(rng.choice(FUNCS)), "FUNC"), ("(", "PUN")]
        for k in range(rng.randint(0, 3)):
            if k:
                toks.append((",", "PUN"))
            toks += atom()
        toks.append((")", "PUN"))
        return toks

    def expr():
        toks = call() if rng.rand() < 0.35 else atom()
        for _ in range(rng.randint(0, 2)):
            toks.append((str(rng.choice(OPS1 + OPS2)), "OP"))
            toks += call() if rng.rand() < 0.25 else atom()
        return toks

    def statement():
        r = rng.rand()
        if r < 0.30:                  # assignment: VAR = expr
            return [(str(rng.choice(IDENTS)), "VAR"), ("=", "OP")] + expr()
        if r < 0.50:                  # bare call
            return call()
        if r < 0.68:                  # for VAR in call :
            return [("for", "KW"), (str(rng.choice(IDENTS)), "VAR"),
                    ("in", "KW")] + call() + [(":", "PUN")]
        if r < 0.82:                  # if expr OP atom :
            return [("if", "KW")] + expr() + \
                   [(str(rng.choice(OPS1)), "OP")] + atom() + [(":", "PUN")]
        if r < 0.92:                  # return expr
            return [("return", "KW")] + expr()
        # comment: # word word ...   (words overlap with code identifiers)
        toks = [("#", "COM")]
        for _ in range(rng.randint(2, 5)):
            toks.append((str(rng.choice(CWORDS)), "COM"))
        return toks

    data = []
    for _ in range(n):
        stmt = statement()
        data.append(([t for t, _ in stmt], [l for _, l in stmt]))
    return data


def macro_f1(gold, pred):
    g, p, f1s = np.array(gold), np.array(pred), []
    for c in LABELS:
        tp = np.sum((p == c) & (g == c))
        fp = np.sum((p == c) & (g != c))
        fn = np.sum((p != c) & (g == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


# ------------------------------- demo helpers --------------------------------

COLORS = {"KW": 95, "FUNC": 94, "VAR": 37, "NUM": 96,
          "STR": 92, "OP": 93, "PUN": 90, "COM": 90}


def tokenize_line(line):
    # Minimal hand-written tokenizer for the demo (no regex / no libraries).
    line = line.strip()
    toks, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c == " " or c == "\t":
            i += 1
        elif c == "#":                       # rest of line is a comment
            toks.append("#")
            toks += line[i + 1:].split()
            break
        elif c == '"':
            j = i + 1
            while j < n and line[j] != '"':
                j += 1
            toks.append(line[i:j + 1])
            i = j + 1
        elif c.isdigit():
            j = i
            while j < n and (line[j].isdigit() or line[j] == "."):
                j += 1
            toks.append(line[i:j])
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (line[j].isalnum() or line[j] == "_"):
                j += 1
            toks.append(line[i:j])
            i = j
        elif line[i:i + 2] in OPS2:
            toks.append(line[i:i + 2])
            i += 2
        else:
            toks.append(c)
            i += 1
    return toks


def highlight(model, code):
    out = []
    for line in code.strip("\n").split("\n"):
        toks = tokenize_line(line)
        if not toks:
            out.append("")
            continue
        labs = model.predict(toks)
        out.append("  ".join("\x1b[%dm%s\x1b[0m/%s"
                             % (COLORS[l], t, l) for t, l in zip(toks, labs)))
    return "\n".join(out)


if __name__ == "__main__":
    np.random.seed(0)

    data = make_corpus(n=280, seed=0)
    idx = np.random.permutation(len(data))
    split = int(0.75 * len(data))
    train = [data[i] for i in idx[:split]]
    test = [data[i] for i in idx[split:]]

    model = SyntaxHighlighter(iters=250, lr=0.5).fit(train)

    gold = [l for _, labs in test for l in labs]
    pred = [l for toks, _ in test for l in model.predict(toks)]
    acc = np.mean(np.array(gold) == np.array(pred))
    f1 = macro_f1(gold, pred)

    # Baseline 1: predict the single most frequent training label everywhere.
    tr_labs = [l for _, labs in train for l in labs]
    majority = max(LABELS, key=tr_labs.count)
    b_acc = np.mean(np.array(gold) == majority)
    b_f1 = macro_f1(gold, [majority] * len(gold))

    # Baseline 2: memorise most common label per token string (back off to
    # majority).  Fails on shared FUNC/VAR names and comment words -> shows
    # that the model's *context* is what wins, not lookup.
    from collections import Counter, defaultdict
    seen = defaultdict(Counter)
    for toks, labs in train:
        for t, l in zip(toks, labs):
            seen[t.lower()][l] += 1
    lut_pred = [seen[t.lower()].most_common(1)[0][0] if t.lower() in seen
                else majority for toks, _ in test for t in toks]
    l_acc = np.mean(np.array(gold) == np.array(lut_pred))
    l_f1 = macro_f1(gold, lut_pred)

    print("Snippets: %d   Train: %d   Test: %d   Test tokens: %d"
          % (len(data), len(train), len(test), len(gold)))
    print("Feature vocab: %d   Classes: %d" % (len(model.fidx), len(LABELS)))
    print("-" * 62)
    print("Softmax tagger    accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Token-lookup base accuracy: %.4f   macro-F1: %.4f" % (l_acc, l_f1))
    print("Majority '%s' base  accuracy: %.4f   macro-F1: %.4f"
          % (majority, b_acc, b_f1))
    print("-" * 62)
    demo = ("# compute the total score\n"
            "total = 0\n"
            "for item in range ( count ) :\n"
            "total = total + score ( item )\n"
            "print ( total )")
    print("Highlighted demo (token/CATEGORY):")
    print(highlight(model, demo))
    print("-" * 62)
    print("Softmax tagger beats both baselines: %s"
          % (acc > l_acc and acc > b_acc and f1 > l_f1 and f1 > b_f1))
