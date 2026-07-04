import numpy as np


def tokenize(text):
    return text.lower().split()


class MultinomialNaiveBayes:
    """Bag-of-words spam classifier trained by counting, scored in log-space."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha          # Laplace smoothing (unseen words != 0 prob)

    def _vectorize(self, docs):
        # Word-count matrix over the fitted vocabulary; OOV words dropped.
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] += 1.0
        return X

    def fit(self, docs, y):
        y = np.asarray(y)
        # Vocabulary = every word seen in training.
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        X = self._vectorize(docs)
        V = len(self.vocab)

        self.classes = np.unique(y)
        self.log_prior = np.zeros(len(self.classes))
        self.log_lik = np.zeros((len(self.classes), V))
        for c in self.classes:
            Xc = X[y == c]
            self.log_prior[c] = np.log(Xc.shape[0] / X.shape[0])
            # P(word|class) = (count + alpha) / (total + alpha*V), in log-space.
            counts = Xc.sum(0) + self.alpha
            self.log_lik[c] = np.log(counts / counts.sum())
        return self

    def predict(self, docs):
        X = self._vectorize(docs)
        # log P(class|doc) proportional-to log_prior + sum counts * log P(word|class)
        scores = self.log_prior[None, :] + X @ self.log_lik.T
        return self.classes[np.argmax(scores, axis=1)]


def make_emails(n=600, seed=0):
    # Synthetic corpus: spam and ham draw from overlapping vocabularies but
    # with different word frequencies, so the planted signal is recoverable.
    rng = np.random.RandomState(seed)
    spam_words = ["free", "winner", "cash", "offer", "click", "prize",
                  "cheap", "money", "buy", "urgent", "guarantee", "deal"]
    ham_words = ["meeting", "project", "report", "lunch", "team", "schedule",
                 "review", "please", "thanks", "update", "document", "call"]
    shared = ["the", "you", "today", "this", "and", "for", "our", "now"]

    docs, labels = [], []
    for _ in range(n):
        spam = rng.rand() < 0.4                      # 40% spam, 60% ham
        topic = spam_words if spam else ham_words
        length = rng.randint(8, 16)
        # Mostly topic words + some shared filler -> overlapping but separable.
        n_topic = int(length * 0.65)
        words = list(rng.choice(topic, n_topic)) + \
            list(rng.choice(shared, length - n_topic))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(1 if spam else 0)             # 1 = spam, 0 = ham
    return docs, np.array(labels)


def metrics(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)

    docs, y = make_emails(n=600, seed=0)
    # Held-out split: train on 70%, test on the rest.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Xtr = [docs[i] for i in tr]
    Xte = [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    clf = MultinomialNaiveBayes(alpha=1.0).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc, prec, rec, f1 = metrics(yte, pred)

    # Majority-class baseline: predict whichever label is more common in train.
    majority = int(np.round(ytr.mean()))
    base_pred = np.full_like(yte, majority)
    base_acc, _, _, base_f1 = metrics(yte, base_pred)

    print("Emails: %d   Train: %d   Test: %d   Spam rate: %.2f"
          % (len(docs), len(tr), len(te), y.mean()))
    print("Vocabulary size: %d" % len(clf.vocab))
    print("-" * 56)
    print("Naive Bayes  accuracy: %.4f   F1(spam): %.4f" % (acc, f1))
    print("Majority     accuracy: %.4f   F1(spam): %.4f" % (base_acc, base_f1))
    print("-" * 56)
    for text in ["free cash prize click now", "team meeting report today please"]:
        p = clf.predict([text])[0]
        print("  '%s' -> %s" % (text, "SPAM" if p else "HAM"))
    print("-" * 56)
    print("Naive Bayes beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
