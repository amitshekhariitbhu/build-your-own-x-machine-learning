import numpy as np
import re

# WhatsApp exports look like:  "12/03/23, 10:15 - Alice: this is great!"
# We parse that line format, then classify each message's sentiment with a
# from-scratch multinomial Naive Bayes over bag-of-words features.

LINE_RE = re.compile(r"^\d{2}/\d{2}/\d{2}, \d{2}:\d{2} - ([^:]+): (.*)$")


def parse_chat(lines):
    """Split raw WhatsApp lines into (sender, message) pairs; skip system lines."""
    parsed = []
    for ln in lines:
        m = LINE_RE.match(ln)
        if m:
            parsed.append((m.group(1), m.group(2)))
    return parsed


def tokenize(text):
    # Lowercase, keep word chars and emoticons-as-words; drop punctuation.
    return re.findall(r"[a-z']+", text.lower())


class CountVectorizer:
    """Bag-of-words: learn a vocabulary, map messages to term-count rows."""

    def fit(self, docs):
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)          # ignore words unseen in training
                if j is not None:
                    X[i, j] += 1.0
        return X


class MultinomialNaiveBayes:
    """NB from scratch: class priors + Laplace-smoothed word likelihoods,
    scored in log-space so long messages don't underflow."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        n, v = X.shape
        self.classes = np.unique(y)
        self.log_prior = np.zeros(len(self.classes))
        self.log_likelihood = np.zeros((len(self.classes), v))
        for i, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[i] = np.log(len(Xc) / n)
            counts = Xc.sum(0) + self.alpha            # per-word counts + smoothing
            self.log_likelihood[i] = np.log(counts / counts.sum())
        return self

    def predict(self, X):
        scores = self.log_prior[None, :] + X @ self.log_likelihood.T
        return self.classes[np.argmax(scores, axis=1)]


def make_chat(n=1000, seed=0):
    # Synthetic WhatsApp export: each message is drawn from a sentiment class
    # (0=negative, 1=positive). Class words are the planted signal; neutral
    # filler + shared chat words add realistic noise.
    rng = np.random.RandomState(seed)
    pos = ["love", "great", "awesome", "happy", "thanks", "perfect", "yay",
           "amazing", "haha", "good", "nice", "excited", "best", "wonderful"]
    neg = ["hate", "terrible", "sad", "angry", "worst", "annoying", "ugh",
           "awful", "sorry", "bad", "boring", "tired", "sick", "disappointed"]
    filler = ["the", "i", "you", "we", "is", "it", "to", "today", "now",
              "meeting", "lunch", "later", "ok", "call", "here", "home"]
    senders = ["Alice", "Bob", "Carol", "Dave"]

    lines, labels = [], []
    for k in range(n):
        c = rng.randint(2)
        length = rng.randint(4, 10)
        n_sig = max(1, int(length * 0.5))              # ~half sentiment words
        bank = pos if c == 1 else neg
        words = list(rng.choice(bank, n_sig)) + \
            list(rng.choice(filler, length - n_sig))
        rng.shuffle(words)
        sender = senders[rng.randint(len(senders))]
        ts = "%02d/%02d/23, %02d:%02d" % (
            rng.randint(1, 28), rng.randint(1, 12), rng.randint(0, 24), rng.randint(0, 60))
        lines.append("%s - %s: %s" % (ts, sender, " ".join(words)))
        labels.append(c)
    return lines, np.array(labels)


def macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return np.mean(f1s)


if __name__ == "__main__":
    np.random.seed(0)

    raw_lines, y = make_chat(n=1000, seed=0)
    pairs = parse_chat(raw_lines)                       # (sender, message) pairs
    senders = [s for s, _ in pairs]
    msgs = [m for _, m in pairs]
    labels = ["Negative", "Positive"]

    # Held-out split: train on 70% of messages, test on the rest.
    idx = np.random.permutation(len(msgs))
    split = int(0.7 * len(msgs))
    tr, te = idx[:split], idx[split:]
    tr_docs, te_docs = [msgs[i] for i in tr], [msgs[i] for i in te]
    ytr, yte = y[tr], y[te]

    vec = CountVectorizer().fit(tr_docs)
    Xtr, Xte = vec.transform(tr_docs), vec.transform(te_docs)
    clf = MultinomialNaiveBayes(alpha=1.0).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, clf.classes)

    # Majority-class baseline: always predict the most common training label.
    majority = np.bincount(ytr).argmax()
    base_acc = np.mean(np.full_like(yte, majority) == yte)
    base_f1 = macro_f1(yte, np.full_like(yte, majority), clf.classes)

    print("Parsed messages: %d   Train: %d   Test: %d   Senders: %d"
          % (len(msgs), len(tr), len(te), len(set(senders))))
    print("Vocabulary size: %d" % len(vec.vocab))
    print("-" * 60)
    print("Naive Bayes    accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority class accuracy: %.4f   macro-F1: %.4f" % (base_acc, base_f1))
    print("-" * 60)
    demo = ["10/05/23, 09:12 - Alice: haha this is awesome love it thanks",
            "11/05/23, 18:44 - Bob: ugh worst meeting so boring and annoying",
            "12/05/23, 20:01 - Carol: lunch today ok call me later home"]
    for ln in parse_chat(demo):
        c = clf.predict(vec.transform([ln[1]]))[0]
        print("  %-6s: '%s' -> %s" % (ln[0], ln[1][:30], labels[c]))
    print("-" * 60)
    # Per-sender mood: fraction of that sender's messages predicted positive.
    all_pred = clf.predict(vec.transform(msgs))
    for s in sorted(set(senders)):
        mask = np.array([snd == s for snd in senders])
        print("  %-6s positivity: %.2f" % (s, all_pred[mask].mean()))
    print("-" * 60)
    print("Naive Bayes beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
