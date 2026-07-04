import numpy as np

# Calendar scheduling from natural language.
# Two from-scratch models turn a request like "book a call with Emma Lee on
# Friday at 3 pm for 30min" into a structured event.  (1) A multinomial Naive
# Bayes classifier picks the INTENT (schedule / cancel / reschedule / query);
# (2) a softmax slot tagger labels every token (BIO) as PERSON / DAY / TIME /
# DUR so the fields can be decoded.  Both are trained by hand on templated
# synthetic text -- no library does the classification for us.

SLOTS = ["O", "B-PER", "I-PER", "B-DAY", "B-TIME", "I-TIME", "B-DUR"]
SIDX = {s: i for i, s in enumerate(SLOTS)}
KEY = {"PER": "per", "DAY": "day", "TIME": "time", "DUR": "dur"}
D = 2048  # hashed feature dimension


def fnv(s):
    # Deterministic string hash (FNV-1a) so featurisation is reproducible.
    h = 2166136261
    for c in s:
        h = ((h ^ ord(c)) * 16777619) & 0xFFFFFFFF
    return h


def tok_feats(words, i):
    # Token features: form, affixes, shape and neighbour cues.  PERSON vs DAY
    # (both capitalised) is separated by context ("with" vs "on"/"to"), not
    # case; the prev-token shape cues (pcap/pdig) split B- from I- spans.
    w = words[i]; lw = w.lower()
    pw = words[i - 1] if i > 0 else "<s>"
    nw = words[i + 1] if i + 1 < len(words) else "</s>"
    return ["bias", "w=" + lw, "suf3=" + lw[-3:], "pre2=" + lw[:2],
            "dig=" + str(any(c.isdigit() for c in w)),
            "colon=" + str(":" in w),
            "ap=" + str(lw in ("am", "pm", "noon")),
            "cap=" + str(w[:1].isupper()),
            "pw=" + pw.lower(), "nw=" + nw.lower(),
            "pcap=" + str(pw[:1].isupper()),
            "pdig=" + str(any(c.isdigit() for c in pw))]


def to_matrix(feat_rows):
    # Multi-hot feature matrix via hashing (collisions tolerated).
    X = np.zeros((len(feat_rows), D))
    for r, feats in enumerate(feat_rows):
        for f in feats:
            X[r, fnv(f) % D] = 1.0
    return X


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(1, keepdims=True)


class SlotTagger:
    """Multiclass logistic-regression token tagger (full-batch gradient descent)."""

    def fit(self, X, y, iters=120, lr=0.5, l2=1e-4):
        n, d = X.shape
        Y = np.eye(len(SLOTS))[y]
        self.W = np.zeros((d, len(SLOTS)))
        for _ in range(iters):
            grad = X.T @ (softmax(X @ self.W) - Y) / n + l2 * self.W
            self.W -= lr * grad
        return self

    def predict(self, words):
        X = to_matrix([tok_feats(words, i) for i in range(len(words))])
        return [SLOTS[k] for k in np.argmax(X @ self.W, axis=1)]


class IntentNB:
    """Multinomial Naive Bayes over bag-of-words with Laplace smoothing."""

    def fit(self, docs, labels):
        self.classes = sorted(set(labels))
        self.vi = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        V = len(self.vi)
        self.prior, self.ll = {}, {}
        for c in self.classes:
            cnt, k = np.ones(V), 0  # Laplace: unseen word != zero prob
            for d, l in zip(docs, labels):
                if l == c:
                    k += 1
                    for w in d:
                        cnt[self.vi[w]] += 1
            self.ll[c] = np.log(cnt / cnt.sum())
            self.prior[c] = np.log(k / len(docs))
        return self

    def predict(self, doc):
        best, bs = self.classes[0], -1e18
        for c in self.classes:
            s = self.prior[c] + sum(self.ll[c][self.vi[w]] for w in doc if w in self.vi)
            if s > bs:
                bs, best = s, c
        return best


def decode_fields(words, tags):
    # Collapse a BIO tag sequence into the calendar event's filled fields.
    f = {"per": "", "day": "", "time": "", "dur": ""}
    i, n = 0, len(tags)
    while i < n:
        if tags[i].startswith("B-"):
            typ, s = tags[i][2:], i
            i += 1
            while i < n and tags[i] == "I-" + typ:
                i += 1
            f[KEY[typ]] = " ".join(w.lower() for w in words[s:i])
        else:
            i += 1
    return f


def spans(tags):
    out, i, n = set(), 0, len(tags)
    while i < n:
        if tags[i].startswith("B-"):
            typ, s = tags[i][2:], i
            i += 1
            while i < n and tags[i] == "I-" + typ:
                i += 1
            out.add((s, i, typ))
        else:
            i += 1
    return out


def make_data(n=420, seed=0):
    # Templated requests with planted intent + BIO slots.  Days and names are
    # both capitalised, so slot type is only recoverable from context words.
    rng = np.random.RandomState(seed)
    first = ["Alice", "Bob", "Emma", "David", "Sara", "John", "Maya", "Omar"]
    last = ["Smith", "Jones", "Lee", "Khan", "Brown", "Patel"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow", "today"]
    durs = ["30min", "45min", "1h", "2hours", "1hour", "15min"]
    templates = {
        "schedule": [["schedule", "a", "meeting", "with", "PER", "on", "DAY", "at", "TIME"],
                     ["book", "a", "call", "with", "PER", "on", "DAY", "for", "DUR"],
                     ["set", "up", "a", "sync", "with", "PER", "on", "DAY"]],
        "cancel": [["cancel", "my", "meeting", "with", "PER", "on", "DAY"],
                   ["cancel", "the", "call", "on", "DAY", "at", "TIME"],
                   ["drop", "the", "sync", "with", "PER"]],
        "reschedule": [["move", "my", "meeting", "with", "PER", "to", "DAY", "at", "TIME"],
                       ["reschedule", "the", "call", "to", "DAY"],
                       ["push", "the", "sync", "with", "PER", "to", "TIME"]],
        "query": [["what", "meetings", "do", "i", "have", "on", "DAY"],
                  ["when", "is", "my", "call", "with", "PER"],
                  ["do", "i", "have", "anything", "on", "DAY", "at", "TIME"]],
    }
    intents = list(templates)

    def a_time():
        r = rng.rand()
        if r < 0.35:
            return [rng.choice(["9am", "3pm", "11am", "5pm"])]
        if r < 0.6:
            return [rng.choice(["9", "3", "11", "2"]), rng.choice(["am", "pm"])]
        if r < 0.8:
            return [rng.choice(["10:30", "14:00", "09:15"])]
        return ["noon"]

    def fill(tok):
        if tok == "PER":
            p = [rng.choice(first)] + ([rng.choice(last)] if rng.rand() < 0.5 else [])
            return p, "PER"
        if tok == "DAY":
            return [rng.choice(days)], "DAY"
        if tok == "TIME":
            return a_time(), "TIME"
        if tok == "DUR":
            return [rng.choice(durs)], "DUR"
        return [tok], None

    data = []
    for _ in range(n):
        intent = intents[rng.randint(len(intents))]
        tmpl = templates[intent][rng.randint(len(templates[intent]))]
        words, tags = [], []
        for tok in tmpl:
            toks, typ = fill(tok)
            for j, wd in enumerate(toks):
                words.append(wd)
                tags.append("O" if typ is None else ("B-" if j == 0 else "I-") + typ)
        data.append({"words": words, "tags": tags, "intent": intent,
                     "fields": decode_fields(words, tags)})
    return data


if __name__ == "__main__":
    np.random.seed(0)
    data = make_data(n=420, seed=0)
    split = int(0.75 * len(data))
    train, test = data[:split], data[split:]

    # ---- train intent classifier ----
    nb = IntentNB().fit([d["words"] for d in train], [d["intent"] for d in train])

    # ---- train slot tagger ----
    rows, ys = [], []
    for d in train:
        for i in range(len(d["words"])):
            rows.append(tok_feats(d["words"], i))
            ys.append(SIDX[d["tags"][i]])
    tagger = SlotTagger().fit(to_matrix(rows), np.array(ys))

    # ---- evaluate on held-out split ----
    labels = [d["intent"] for d in train]
    maj = max(set(labels), key=labels.count)      # majority-intent baseline
    empty = {"per": "", "day": "", "time": "", "dur": ""}
    nt = len(test)
    ihit = ehit = b_ihit = b_ehit = tok_tot = tok_hit = 0
    G, P = set(), set()
    for k, d in enumerate(test):
        pi = nb.predict(d["words"])
        pt = tagger.predict(d["words"])
        pf = decode_fields(d["words"], pt)
        ihit += pi == d["intent"]
        ehit += pi == d["intent"] and pf == d["fields"]
        b_ihit += maj == d["intent"]
        b_ehit += maj == d["intent"] and empty == d["fields"]
        for a, b in zip(d["tags"], pt):
            tok_tot += 1; tok_hit += a == b
        G |= {(k,) + s for s in spans(d["tags"])}
        P |= {(k,) + s for s in spans(pt)}
    tp, fp, fn = len(G & P), len(P - G), len(G - P)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print("Sentences: %d   Train: %d   Test: %d   Tokens(test): %d"
          % (len(data), len(train), nt, tok_tot))
    print("-" * 64)
    print("Intent  NaiveBayes acc: %.3f     majority baseline acc: %.3f"
          % (ihit / nt, b_ihit / nt))
    print("Slots   token acc: %.3f   span P: %.3f  R: %.3f  F1: %.3f"
          % (tok_hit / tok_tot, prec, rec, f1))
    print("Event   full exact-match: %.3f     no-slot baseline: %.3f"
          % (ehit / nt, b_ehit / nt))
    print("-" * 64)
    s = "schedule a meeting with Emma Lee on Friday at 3 pm".split()
    fl = decode_fields(s, tagger.predict(s))
    print("Demo request: " + " ".join(s))
    print("  intent = " + nb.predict(s))
    print("  event  = " + ", ".join("%s=%s" % (k, v) for k, v in fl.items() if v))
    print("-" * 64)
    ok = ihit / nt > b_ihit / nt and f1 > 0.5 and ehit / nt > b_ehit / nt
    print("Beats baselines (intent + slots + event): %s" % ok)
