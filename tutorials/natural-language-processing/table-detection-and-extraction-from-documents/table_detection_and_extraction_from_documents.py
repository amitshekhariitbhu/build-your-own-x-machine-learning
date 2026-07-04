import numpy as np

# Table detection & extraction from plain-text documents, from scratch.
#
#   1) DETECTION  - per-line binary classifier (logistic regression trained by
#      gradient descent) that decides "table row" vs "prose", using layout
#      features: wide (multi-space) column gaps, numeric density, token stats.
#   2) EXTRACTION - given the detected table block, recover the cell grid by
#      finding character columns that are whitespace in EVERY row (the column
#      separators) and slicing between them. This keeps multi-word cells like
#      "New York" intact, which a naive whitespace split shreds.


# ----------------------------------------------------------- synthetic data
NAMES  = ["Ann Lee", "Bob Ng", "Cara Ruiz", "Dan Kim", "Eve Osei", "Fay Cruz"]
CITIES = ["New York", "San Jose", "Palo Alto", "Ann Arbor", "El Paso", "Reno"]
SUBJ = ["the team", "our group", "the author", "this report", "the manager"]
VERB = ["describes", "reviews", "summarizes", "presents", "discusses"]
OBJ  = ["the results", "the quarterly plan", "the new policy", "the budget"]
TAIL = ["in detail.", "for the meeting.", "this week.", "before the deadline."]


def make_prose(rng):
    # Natural sentence: single-spaced words, no numbers -> no column layout.
    return " ".join([rng.choice(SUBJ), rng.choice(VERB),
                     rng.choice(OBJ), rng.choice(TAIL)])


def make_table(rng):
    # Ground-truth grid (header + rows). Cells include multi-word and numeric
    # fields so the extractor must respect columns, not just split on spaces.
    grid = [["Name", "City", "Year", "Salary"]]
    for _ in range(rng.randint(3, 7)):
        grid.append([rng.choice(NAMES), rng.choice(CITIES),
                     str(rng.randint(2016, 2025)), "$%dk" % rng.randint(40, 180)])
    return grid


def render_table(grid, rng):
    # Fixed-width columns: each field ljust to (max cell width + gap>=2), so a
    # block of >=2 all-space columns always sits between adjacent fields.
    ncols = len(grid[0])
    widths = [max(len(r[c]) for r in grid) + rng.randint(2, 4) for c in range(ncols)]
    return ["".join(cell.ljust(widths[c]) for c, cell in enumerate(row)).rstrip()
            for row in grid]


def make_document(rng):
    # Alternate prose blocks and rendered tables. Returns the lines, per-line
    # table/prose labels, and the (grid, rendered-lines) of each planted table.
    lines, labels, tables = [], [], []
    for _ in range(rng.randint(2, 4)):
        for _ in range(rng.randint(1, 3)):          # prose block
            lines.append(make_prose(rng)); labels.append(0)
        grid = make_table(rng)                      # table block
        rows = render_table(grid, rng)
        tables.append((grid, rows))
        for r in rows:
            lines.append(r); labels.append(1)
    return lines, labels, tables


# --------------------------------------------------------------- features
def wide_gaps(line):
    # Count interior runs of >=2 spaces (i.e. column separators).
    c, i, L = 0, 0, len(line)
    while i < L:
        if line[i] == " ":
            j = i
            while j < L and line[j] == " ":
                j += 1
            if j - i >= 2 and i > 0 and j < L:
                c += 1
            i = j
        else:
            i += 1
    return c


def line_features(line):
    toks = line.split()
    n = max(len(toks), 1)
    numeric = sum(any(ch.isdigit() for ch in t) for t in toks) / n
    digits = sum(ch.isdigit() for ch in line) / max(len(line), 1)
    avg_len = np.mean([len(t) for t in toks]) if toks else 0.0
    return [len(toks), avg_len, wide_gaps(line), numeric, digits]


# ------------------------------------------------------- logistic regression
class LogReg:
    def fit(self, X, y, lr=0.5, epochs=400):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b)))
            g = p - y
            self.w -= lr * (Xs.T @ g) / n
            self.b -= lr * g.mean()
        return self

    def predict(self, X):
        Xs = (X - self.mu) / self.sd
        return (1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b))) >= 0.5).astype(int)


# ------------------------------------------------------------- extraction
def extract_aligned(lines, min_gap=2):
    # A real column separator is a run of >=min_gap character columns that are
    # a space in EVERY row. Requiring width>=2 is what keeps a multi-word cell
    # ("Palo Alto") together: its internal single space is a 1-wide sliver, not
    # a separator, while the padding between fields is always >=2 wide.
    w = max(len(l) for l in lines)
    P = [l.ljust(w) for l in lines]
    allspace = np.array([[ch == " " for ch in row] for row in P]).all(0)
    is_sep, c = np.zeros(w, bool), 0
    while c < w:
        if allspace[c]:
            s = c
            while c < w and allspace[c]:
                c += 1
            if c - s >= min_gap:
                is_sep[s:c] = True
        else:
            c += 1
    spans, c = [], 0
    while c < w:                                    # fields = non-separator runs
        if not is_sep[c]:
            s = c
            while c < w and not is_sep[c]:
                c += 1
            spans.append((s, c))
        else:
            c += 1
    return [[row[a:b].strip() for a, b in spans] for row in P]


def extract_split(lines):
    return [l.split() for l in lines]               # naive baseline


def row_recovery(pred, true):
    return np.mean([p == t for p, t in zip(pred, true)])


def f1(y, p):
    tp = np.sum((p == 1) & (y == 1)); fp = np.sum((p == 1) & (y == 0))
    fn = np.sum((p == 0) & (y == 1))
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


# ------------------------------------------------------------------- demo
if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    docs = [make_document(rng) for _ in range(80)]      # 80 documents
    split = 56                                          # 70% train / 30% test
    train, test = docs[:split], docs[split:]

    Xtr = np.array([line_features(l) for d in train for l in d[0]])
    ytr = np.array([y for d in train for y in d[1]])
    Xte = np.array([line_features(l) for d in test for l in d[0]])
    yte = np.array([y for d in test for y in d[1]])

    clf = LogReg().fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = np.mean(pred == yte)

    majority = int(round(ytr.mean()))                   # majority-class baseline
    base_acc = np.mean(yte == majority)

    # ---- extraction on the planted tables of the test documents ----------
    align_rec, split_rec, cells_ok, cells_tot = [], [], 0, 0
    for _, _, tables in test:
        for grid, rows in tables:
            g_align, g_split = extract_aligned(rows), extract_split(rows)
            align_rec.append(row_recovery(g_align, grid))
            split_rec.append(row_recovery(g_split, grid))
            for pr, tr in zip(g_align, grid):           # cell-level accuracy
                if len(pr) == len(tr):
                    cells_ok += sum(a == b for a, b in zip(pr, tr))
                cells_tot += len(tr)
    align_rec = float(np.mean(align_rec)); split_rec = float(np.mean(split_rec))
    cell_acc = cells_ok / cells_tot

    print("Documents: %d  Train lines: %d  Test lines: %d  Table-line rate: %.2f"
          % (len(docs), len(ytr), len(yte), yte.mean()))
    print("-" * 62)
    print("DETECTION  logistic-regression accuracy: %.4f   F1: %.4f"
          % (acc, f1(yte, pred)))
    print("DETECTION  majority baseline  accuracy: %.4f" % base_acc)
    print("-" * 62)
    print("EXTRACTION aligned-column exact-row recovery: %.4f  cell-acc: %.4f"
          % (align_rec, cell_acc))
    print("EXTRACTION whitespace-split exact-row recovery: %.4f  (baseline)"
          % split_rec)
    print("-" * 62)
    grid, rows = test[0][2][0]                           # show one recovered table
    print("Sample table row (raw): %r" % rows[1])
    print("Aligned cells        : %s" % extract_aligned(rows)[1])
    print("Whitespace cells     : %s" % extract_split(rows)[1])
    print("-" * 62)
    ok = acc > base_acc and cell_acc > 0.99 and align_rec > split_rec + 0.3
    print("SUCCESS" if ok else "FAIL")
