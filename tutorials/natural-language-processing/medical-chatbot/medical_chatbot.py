import numpy as np


def tokenize(text):
    return text.lower().replace(",", "").replace(".", "").split()


class MedicalChatBot:
    """Symptom-triage chatbot built on a from-scratch multinomial Naive Bayes.

    Each condition is a class. From training messages we learn, per condition,
    the smoothed probability of every symptom word (a word-likelihood table) and
    a class prior. A patient message is classified by log-posterior:
        log P(cond|msg) = log P(cond) + sum_word count * log P(word|cond)
    The winning condition selects a canned triage reply, so 'understanding' the
    patient reduces to classifying their symptoms into the right condition."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha        # Laplace smoothing (unseen symptom words)

    def _counts(self, docs):
        # Bag-of-words count matrix over the fitted symptom vocabulary.
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] += 1.0
        return X

    def fit(self, docs, labels, replies):
        self.classes = np.unique(labels)
        self.replies = replies                      # condition -> triage reply
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        X = self._counts(docs)
        labels = np.asarray(labels)

        V = len(self.vocab)
        self.log_prior = np.zeros(len(self.classes))
        self.log_like = np.zeros((len(self.classes), V))
        for k, c in enumerate(self.classes):
            Xc = X[labels == c]
            self.log_prior[k] = np.log(len(Xc) / len(docs))
            # Per-word likelihood = (word count in class + alpha) / (class words + alpha*V)
            wc = Xc.sum(0) + self.alpha
            self.log_like[k] = np.log(wc / wc.sum())
        return self

    def _log_posterior(self, docs):
        X = self._counts(docs)
        # log-posterior (up to constant) = prior + counts . log-likelihood
        return self.log_prior[None, :] + X @ self.log_like.T

    def predict(self, docs):
        return self.classes[np.argmax(self._log_posterior(docs), axis=1)]

    def respond(self, text):
        cond = self.predict([text])[0]
        return cond, self.replies[cond]


def make_messages(seed=0, per_class=90):
    # Synthetic patient messages with planted conditions. Each condition owns a
    # bank of characteristic symptom words; messages sample a few of them and mix
    # in shared filler ("i", "have", "feeling") so classes overlap yet separate.
    rng = np.random.RandomState(seed)
    symptoms = {
        "flu":       ["fever", "chills", "body", "ache", "cough", "sore",
                      "throat", "fatigue", "headache", "congestion"],
        "migraine":  ["headache", "throbbing", "nausea", "light", "sensitivity",
                      "aura", "vision", "dizzy", "temple", "pounding"],
        "allergy":   ["sneezing", "itchy", "eyes", "runny", "nose", "rash",
                      "hives", "watery", "congestion", "wheezing"],
        "food_pois": ["vomiting", "diarrhea", "stomach", "cramps", "nausea",
                      "dehydration", "fever", "abdominal", "pain", "weakness"],
        "asthma":    ["wheezing", "shortness", "breath", "chest", "tightness",
                      "cough", "breathing", "gasping", "airway", "trouble"],
    }
    replies = {
        "flu":       "Likely flu. Rest, hydrate, and take fever medication; see a doctor if it persists.",
        "migraine":  "Sounds like a migraine. Rest in a dark quiet room and consider pain relief.",
        "allergy":   "Probably an allergy. Try an antihistamine and avoid known triggers.",
        "food_pois": "Possible food poisoning. Hydrate with fluids; seek care if symptoms are severe.",
        "asthma":    "Asthma-like symptoms. Use your inhaler and seek urgent care if breathing worsens.",
    }
    filler = ["i", "have", "been", "feeling", "with", "a", "and", "some",
              "really", "since", "today", "my", "lot", "of"]

    docs, labels = [], []
    for cond, bank in symptoms.items():
        for _ in range(per_class):
            n_sym = rng.randint(3, 6)               # 3-5 symptom words per message
            words = list(rng.choice(bank, n_sym, replace=False))
            words += list(rng.choice(filler, rng.randint(3, 7)))
            rng.shuffle(words)
            docs.append(" ".join(words))
            labels.append(cond)
    return docs, np.array(labels), replies


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

    docs, y, replies = make_messages(seed=0)
    # Held-out split: train NB on 70%, evaluate on unseen 30%.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Xtr, Xte = [docs[i] for i in tr], [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    bot = MedicalChatBot().fit(Xtr, ytr, replies)
    pred = bot.predict(Xte)
    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, bot.classes)

    # Majority-class baseline: always guess the most common training condition.
    vals, cnts = np.unique(ytr, return_counts=True)
    majority = vals[np.argmax(cnts)]
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(yte == majority)
    base_f1 = macro_f1(yte, base_pred, bot.classes)

    print("Messages: %d   Train: %d   Test: %d   Conditions: %d"
          % (len(docs), len(tr), len(te), len(replies)))
    print("Symptom vocabulary size: %d" % len(bot.vocab))
    print("-" * 62)
    print("Naive Bayes  accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority     accuracy: %.4f   macro-F1: %.4f  (always '%s')"
          % (base_acc, base_f1, majority))
    print("-" * 62)
    for q in ["i have a fever chills and a sore throat",
              "throbbing headache with nausea and light sensitivity",
              "sneezing runny nose and itchy watery eyes",
              "wheezing chest tightness and shortness of breath"]:
        cond, reply = bot.respond(q)
        print("  patient: %s" % q)
        print("     -> [%s] %s" % (cond, reply))
    print("-" * 62)
    print("Naive Bayes beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
