import numpy as np


def tokenize(text):
    return text.lower().split()


def email_tokens(subject, body, subject_weight=3):
    # Subject words carry the strongest signal -> repeat them so a TF-IDF
    # count model weighs the subject line more than the longer body.
    return tokenize(subject) * subject_weight + tokenize(body)


class EmailOrganizer:
    """Sorts incoming email into folders with a from-scratch TF-IDF +
    multinomial Naive Bayes model. Everything (vocabulary, IDF weights,
    class priors, per-word likelihoods) is estimated from the training mail."""

    def __init__(self, alpha=1.0, subject_weight=3):
        self.alpha = alpha                    # Laplace smoothing for unseen words
        self.subject_weight = subject_weight

    def _docs(self, emails):
        return [email_tokens(s, b, self.subject_weight) for s, b in emails]

    def _fit_idf(self, tok_docs):
        # Vocabulary = every word seen in training mail.
        self.vocab = {}
        for toks in tok_docs:
            for w in toks:
                self.vocab.setdefault(w, len(self.vocab))
        # Document frequency -> smoothed inverse-document-frequency weights.
        df = np.zeros(len(self.vocab))
        for toks in tok_docs:
            for w in set(toks):
                df[self.vocab[w]] += 1.0
        n = len(tok_docs)
        self.idf = np.log((1.0 + n) / (1.0 + df)) + 1.0

    def _vectorize(self, tok_docs):
        # TF-IDF weighted term counts; out-of-vocabulary words are dropped.
        X = np.zeros((len(tok_docs), len(self.vocab)))
        for i, toks in enumerate(tok_docs):
            for w in toks:
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] += 1.0
        return X * self.idf[None, :]

    def fit(self, emails, folders):
        y = np.asarray(folders)
        tok = self._docs(emails)
        self._fit_idf(tok)
        X = self._vectorize(tok)
        self.classes = np.unique(y)
        V = X.shape[1]
        self.log_prior = np.zeros(len(self.classes))
        self.log_lik = np.zeros((len(self.classes), V))
        for k, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[k] = np.log(Xc.shape[0] / X.shape[0])
            # P(word|folder) = (weight + alpha) / (total + alpha*V), log-space.
            counts = Xc.sum(0) + self.alpha
            self.log_lik[k] = np.log(counts / counts.sum())
        return self

    def predict(self, emails):
        X = self._vectorize(self._docs(emails))
        # log P(folder|email) proportional-to prior + evidence . likelihood.
        scores = self.log_prior[None, :] + X @ self.log_lik.T
        return self.classes[np.argmax(scores, axis=1)]


def make_emails(n=700, seed=0):
    # Synthetic inbox: each folder draws mostly from its own topic vocabulary
    # plus shared filler, so the planted folder signal is recoverable but noisy.
    rng = np.random.RandomState(seed)
    folders = {
        0: ["meeting", "project", "deadline", "report", "review",
            "presentation", "client", "team", "agenda", "schedule"],
        1: ["invoice", "payment", "bank", "statement", "transaction",
            "balance", "account", "tax", "refund", "billing"],
        2: ["flight", "hotel", "booking", "reservation", "itinerary",
            "ticket", "departure", "trip", "boarding", "gate"],
        3: ["sale", "discount", "offer", "deal", "coupon", "free",
            "shop", "limited", "clearance", "save"],
        4: ["dinner", "family", "birthday", "weekend", "photos",
            "friend", "movie", "party", "vacation", "hey"],
    }
    shared = ["the", "you", "please", "this", "and", "for", "your",
              "we", "on", "a", "to", "will"]
    names = list(folders.keys())
    emails, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        topic = folders[c]
        subject = list(rng.choice(topic, rng.randint(2, 5)))    # short, topical
        blen = rng.randint(10, 18)
        n_topic = int(blen * 0.55)                              # 55% topical body
        body = list(rng.choice(topic, n_topic)) + \
            list(rng.choice(shared, blen - n_topic))
        rng.shuffle(body)
        emails.append((" ".join(subject), " ".join(body)))
        labels.append(c)
    return emails, np.array(labels)


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

    emails, y = make_emails(n=700, seed=0)
    folder_names = ["Work", "Finance", "Travel", "Promotions", "Personal"]

    # Held-out split: train on 70% of the inbox, test on the rest.
    idx = np.random.permutation(len(emails))
    split = int(0.7 * len(emails))
    tr, te = idx[:split], idx[split:]
    Etr = [emails[i] for i in tr]
    Ete = [emails[i] for i in te]
    ytr, yte = y[tr], y[te]

    org = EmailOrganizer(alpha=1.0, subject_weight=3).fit(Etr, ytr)
    pred = org.predict(Ete)
    acc = np.mean(pred == yte)
    f1 = macro_f1(yte, pred, org.classes)

    # Majority-folder baseline: dump everything into the most common folder.
    majority = np.bincount(ytr).argmax()
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    base_f1 = macro_f1(yte, base_pred, org.classes)

    print("Emails: %d   Train: %d   Test: %d   Folders: %d"
          % (len(emails), len(tr), len(te), len(folder_names)))
    print("Vocabulary size: %d" % len(org.vocab))
    print("-" * 60)
    print("TF-IDF Naive Bayes  accuracy: %.4f   macro-F1: %.4f" % (acc, f1))
    print("Majority folder     accuracy: %.4f   macro-F1: %.4f"
          % (base_acc, base_f1))
    print("-" * 60)
    inbox = [("quarterly report review", "please send the agenda before the client meeting"),
             ("your invoice is ready", "the payment statement shows your account balance"),
             ("flight booking confirmed", "your hotel reservation and boarding gate itinerary"),
             ("weekend sale 50 off", "limited offer shop the clearance deal and save"),
             ("birthday dinner plans", "hey the family movie and party this weekend")]
    for subj, body in inbox:
        c = org.predict([(subj, body)])[0]
        print("  '%s' -> %s" % (subj[:28].ljust(28), folder_names[c]))
    print("-" * 60)
    print("Organizer beats majority baseline: %s"
          % (acc > base_acc and f1 > base_f1))
