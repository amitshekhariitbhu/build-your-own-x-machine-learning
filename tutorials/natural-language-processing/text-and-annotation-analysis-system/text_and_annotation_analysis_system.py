import numpy as np

# Text and Annotation Analysis System
# -----------------------------------
# Two coupled from-scratch pieces over bag-of-words counts:
#   1. DOCUMENT ANALYSIS  -- a Multinomial Naive Bayes classifier assigns each
#      document a topic (trained by counting, scored in log-space with Laplace
#      smoothing).
#   2. TOKEN ANNOTATION   -- each content word is annotated with the topic it
#      most strongly signals, read straight off the learned class-conditional
#      word weights (argmax_c log P(word|c)).
# Correctness is proved on a held-out split: document accuracy vs the majority
# baseline, and annotation accuracy on topic-bearing words vs random (1/K).

TOPICS = ["sports", "tech", "finance", "health"]

TOPIC_VOCAB = {
    "sports":  ["game", "team", "coach", "player", "score", "match",
                "league", "season", "goal", "stadium"],
    "tech":    ["software", "chip", "data", "network", "server", "algorithm",
                "code", "device", "cloud", "robot"],
    "finance": ["stock", "market", "bank", "invest", "profit", "loan",
                "bond", "trade", "revenue", "tax"],
    "health":  ["doctor", "patient", "disease", "hospital", "vaccine",
                "therapy", "symptom", "clinic", "surgery", "diet"],
}
FILLER = ["the", "a", "in", "and", "of", "to", "is", "on", "with",
          "for", "this", "that", "will", "was", "as", "at"]


def build_vocab(docs):
    # Vocabulary = every word seen in the training documents.
    vocab = {}
    for d in docs:
        for w in d:
            vocab.setdefault(w, len(vocab))
    return vocab


def vectorize(docs, vocab):
    # Word-count matrix over the fitted vocabulary; OOV words are dropped.
    X = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for w in d:
            j = vocab.get(w)
            if j is not None:
                X[i, j] += 1.0
    return X


class NaiveBayesText:
    """Multinomial Naive Bayes topic classifier with per-word topic weights."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha          # Laplace smoothing (unseen words != 0 prob)

    def fit(self, X, y, n_classes):
        self.log_prior = np.zeros(n_classes)
        V = X.shape[1]
        # feature_log_prob[c, w] = log P(word w | class c).
        self.feature_log_prob = np.zeros((n_classes, V))
        for c in range(n_classes):
            Xc = X[y == c]
            self.log_prior[c] = np.log(Xc.shape[0] / X.shape[0])
            counts = Xc.sum(0) + self.alpha
            self.feature_log_prob[c] = np.log(counts / counts.sum())
        return self

    def predict(self, X):
        # log P(class|doc) proportional-to log_prior + sum counts*log P(word|c).
        scores = self.log_prior[None, :] + X @ self.feature_log_prob.T
        return np.argmax(scores, axis=1)

    def annotate(self, vocab, doc):
        # Tag each content word of a document with its most-signalled topic.
        word_topic = np.argmax(self.feature_log_prob, axis=0)
        out = []
        for w in doc:
            j = vocab.get(w)
            if j is not None:
                out.append((w, TOPICS[word_topic[j]]))
        return out


def make_corpus(n=600, seed=0):
    # Synthetic corpus: each document draws mostly from ONE topic's vocabulary,
    # plus shared filler and a little cross-topic noise, so both the document
    # label and each word's owning topic are recoverable planted structure.
    rng = np.random.RandomState(seed)
    owner = {}                                   # word -> true topic id (-1 filler)
    for ci, t in enumerate(TOPICS):
        for w in TOPIC_VOCAB[t]:
            owner[w] = ci
    for w in FILLER:
        owner[w] = -1

    docs, labels = [], []
    for _ in range(n):
        c = rng.randint(len(TOPICS))
        length = rng.randint(15, 30)
        toks = []
        for _ in range(length):
            r = rng.rand()
            if r < 0.55:                         # on-topic content word
                toks.append(rng.choice(TOPIC_VOCAB[TOPICS[c]]))
            elif r < 0.85:                       # topic-neutral filler
                toks.append(rng.choice(FILLER))
            else:                                # cross-topic noise
                oc = rng.randint(len(TOPICS))
                toks.append(rng.choice(TOPIC_VOCAB[TOPICS[oc]]))
        docs.append(toks)
        labels.append(c)
    return docs, np.array(labels), owner


if __name__ == "__main__":
    np.random.seed(0)

    docs, y, owner = make_corpus(n=600, seed=0)
    split = int(0.75 * len(docs))
    tr_docs, te_docs = docs[:split], docs[split:]
    y_tr, y_te = y[:split], y[split:]

    vocab = build_vocab(tr_docs)
    Xtr, Xte = vectorize(tr_docs, vocab), vectorize(te_docs, vocab)

    model = NaiveBayesText(alpha=1.0).fit(Xtr, y_tr, len(TOPICS))

    # 1) Document analysis: accuracy vs the majority-class baseline.
    pred = model.predict(Xte)
    acc = np.mean(pred == y_te)
    majority = np.bincount(y_tr).argmax()
    base_acc = np.mean(y_te == majority)

    # 2) Token annotation: accuracy on topic-bearing words vs random (1/K).
    word_topic = np.argmax(model.feature_log_prob, axis=0)
    hits = tot = 0
    for w, j in vocab.items():
        o = owner.get(w, -1)
        if o >= 0:
            tot += 1
            hits += int(word_topic[j] == o)
    ann_acc = hits / tot
    rand_ann = 1.0 / len(TOPICS)

    print("Docs: %d  Train: %d  Test: %d  Vocab: %d  Topics: %d"
          % (len(docs), len(tr_docs), len(te_docs), len(vocab), len(TOPICS)))
    print("-" * 60)
    print("Document classification accuracy : %.3f" % acc)
    print("Majority-class baseline          : %.3f" % base_acc)
    print("Token annotation accuracy        : %.3f  (on %d topic words)"
          % (ann_acc, tot))
    print("Random annotation baseline       : %.3f" % rand_ann)
    print("-" * 60)
    sample = te_docs[0]
    print("Demo doc true topic: %-8s predicted: %s"
          % (TOPICS[y_te[0]], TOPICS[pred[0]]))
    seen, shown = set(), []
    for w, t in model.annotate(vocab, sample):
        if w not in seen and owner.get(w, -1) >= 0:
            seen.add(w)
            shown.append("%s/%s" % (w, t))
        if len(shown) >= 6:
            break
    print("Annotated tokens   : " + "  ".join(shown))
    print("-" * 60)
    print("NB beats baselines: %s"
          % (acc > base_acc and ann_acc > rand_ann))
