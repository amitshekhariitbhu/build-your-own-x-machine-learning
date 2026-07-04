import numpy as np

# Named Entity Recognition via an averaged structured perceptron.
# Tokens are tagged with the BIO scheme; each token is classified from
# hand-crafted features (word shape, affixes, neighbours) plus the previous
# predicted tag, and sentences are decoded greedily left-to-right.

TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
TIDX = {t: i for i, t in enumerate(TAGS)}


def word_feats(words, i, prev_tag):
    # Features fired for token i; identical across tags except "prevtag",
    # which injects the sequence/transition signal into the classifier.
    w = words[i]
    lw = w.lower()
    return ["bias",
            "w=" + lw,
            "pre3=" + lw[:3],
            "suf3=" + lw[-3:],
            "cap=" + str(w[:1].isupper()),
            "up=" + str(w.isupper()),
            "prevw=" + (words[i - 1].lower() if i > 0 else "<s>"),
            "nextw=" + (words[i + 1].lower() if i + 1 < len(words) else "</s>"),
            "prevtag=" + prev_tag]


def score(W, feats):
    # Sum the weight vectors (one entry per tag) of the active features.
    s = np.zeros(len(TAGS))
    for f in feats:
        a = W.get(f)
        if a is not None:
            s += a
    return s


class PerceptronNER:
    """Greedy structured perceptron tagger with weight averaging."""

    def __init__(self, epochs=8):
        self.epochs = epochs

    def fit(self, data):
        self.w, u, c = {}, {}, 1  # weights, lazy-average cache, timestamp

        def touch(f):
            if f not in self.w:
                self.w[f] = np.zeros(len(TAGS))
                u[f] = np.zeros(len(TAGS))

        for _ in range(self.epochs):
            for idx in np.random.permutation(len(data)):
                words, gold = data[idx]
                pred_prev = "<start>"
                for i in range(len(words)):
                    # Predict with the *predicted* previous tag (exposure).
                    fp = word_feats(words, i, pred_prev)
                    p = int(np.argmax(score(self.w, fp)))
                    g = TIDX[gold[i]]
                    if p != g:
                        gp = gold[i - 1] if i > 0 else "<start>"
                        fg = word_feats(words, i, gp)  # gold uses gold history
                        for f in fg:
                            touch(f)
                            self.w[f][g] += 1
                            u[f][g] += c
                        for f in fp:
                            touch(f)
                            self.w[f][p] -= 1
                            u[f][p] -= c
                    pred_prev = TAGS[p]
                    c += 1
        # Averaged weights generalise far better than the final ones.
        self.avg = {f: self.w[f] - u[f] / c for f in self.w}
        return self

    def predict(self, words):
        prev, out = "<start>", []
        for i in range(len(words)):
            t = TAGS[int(np.argmax(score(self.avg, word_feats(words, i, prev))))]
            out.append(t)
            prev = t
        return out


def entities(tags):
    # Decode a BIO tag sequence into a set of (start, end, type) spans.
    ents, i, n = set(), 0, len(tags)
    while i < n:
        t = tags[i]
        if t != "O":
            typ, start = t[2:], i
            i += 1
            while i < n and tags[i] == "I-" + typ:
                i += 1
            ents.add((start, i, typ))
        else:
            i += 1
    return ents


def entity_prf(golds, preds):
    G, P = set(), set()
    for k, (g, p) in enumerate(zip(golds, preds)):
        G |= {(k,) + e for e in entities(g)}
        P |= {(k,) + e for e in entities(p)}
    tp, fp, fn = len(G & P), len(P - G), len(G - P)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def make_data(n=400, seed=0):
    # Synthetic corpus: templated sentences with planted PER/LOC/ORG spans.
    # Entity type is recoverable from context words + the entity token itself,
    # not from capitalisation alone (every entity word is capitalised).
    rng = np.random.RandomState(seed)
    first = ["John", "Mary", "Alice", "Bob", "Emma", "David", "Sarah", "Laura"]
    last = ["Smith", "Jones", "Brown", "Lee", "Wang", "Garcia", "Khan", "Patel"]
    loc1 = ["Paris", "London", "Tokyo", "Berlin", "Madrid", "Boston", "Oslo"]
    loc2 = ["New York", "San Diego", "Cape Town", "Hong Kong"]
    org1 = ["Google", "Acme", "Globex", "Initech", "Stark", "Hooli", "Vandelay"]
    org2 = ["Acme Corp", "Globex Inc", "Stark Industries"]
    templates = [["PER", "works", "at", "ORG", "in", "LOC"],
                 ["PER", "visited", "LOC", "last", "year"],
                 ["ORG", "hired", "PER", "as", "ceo"],
                 ["PER", "and", "PER", "met", "in", "LOC"],
                 ["the", "ceo", "of", "ORG", "is", "PER"],
                 ["ORG", "opened", "an", "office", "in", "LOC"],
                 ["PER", "lives", "in", "LOC"]]

    def sample(kind):
        if kind == "PER":
            return [rng.choice(first), rng.choice(last)] if rng.rand() < 0.7 \
                else [rng.choice(first)]
        pool2, pool1 = (loc2, loc1) if kind == "LOC" else (org2, org1)
        return list(str(rng.choice(pool2)).split()) if rng.rand() < 0.3 \
            else [str(rng.choice(pool1))]

    data = []
    for _ in range(n):
        tmpl = templates[rng.randint(len(templates))]
        words, tags = [], []
        for tok in tmpl:
            if tok in ("PER", "LOC", "ORG"):
                for j, wd in enumerate(sample(tok)):
                    words.append(wd)
                    tags.append(("B-" if j == 0 else "I-") + tok)
            else:
                words.append(tok)
                tags.append("O")
        data.append((words, tags))
    return data


if __name__ == "__main__":
    np.random.seed(0)

    data = make_data(n=400, seed=0)
    split = int(0.75 * len(data))
    train, test = data[:split], data[split:]

    model = PerceptronNER(epochs=8).fit(train)

    golds = [g for _, g in test]
    preds = [model.predict(w) for w, _ in test]

    prec, rec, f1 = entity_prf(golds, preds)
    flat_g = [t for g in golds for t in g]
    flat_p = [t for p in preds for t in p]
    tok_acc = np.mean(np.array(flat_g) == np.array(flat_p))

    # Baseline: label every token "O" (the majority tag) -> zero entities.
    base_acc = np.mean(np.array(flat_g) == "O")
    b_prec, b_rec, b_f1 = entity_prf(golds, [["O"] * len(g) for g in golds])

    print("Sentences: %d   Train: %d   Test: %d   Tokens(test): %d"
          % (len(data), len(train), len(test), len(flat_g)))
    print("-" * 60)
    print("Perceptron NER  entity  P: %.3f  R: %.3f  F1: %.3f   tokenAcc: %.3f"
          % (prec, rec, f1, tok_acc))
    print("All-'O' baseline entity  P: %.3f  R: %.3f  F1: %.3f   tokenAcc: %.3f"
          % (b_prec, b_rec, b_f1, base_acc))
    print("-" * 60)
    sent = "Sarah visited London last year".split()
    tagged = model.predict(sent)
    print("Demo: " + "  ".join("%s/%s" % (w, t) for w, t in zip(sent, tagged)))
    print("Extracted: " + ", ".join(
        "%s[%d:%d]" % (typ, s, e) for s, e, typ in sorted(entities(tagged))))
    print("-" * 60)
    print("Perceptron beats all-'O' baseline: %s"
          % (f1 > b_f1 and tok_acc > base_acc))
