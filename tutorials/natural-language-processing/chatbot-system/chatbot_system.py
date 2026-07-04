import numpy as np


def tokenize(text):
    return text.lower().replace("?", "").replace("!", "").split()


class ChatBot:
    """Retrieval-based chatbot: TF-IDF encode each utterance, then answer a
    query by finding its nearest training utterance via cosine similarity.

    The neighbour's intent picks a canned reply, so understanding the user =
    classifying the query into the right intent. TF-IDF down-weights common
    filler words ('i', 'the', 'you') so the intent-bearing words dominate."""

    def __init__(self):
        self.vocab = {}

    def _counts(self, docs):
        # Bag-of-words count matrix over the fitted vocabulary; OOV dropped.
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] += 1.0
        return X

    def _tfidf(self, X):
        # tf = per-doc term frequency; multiply by learned idf, then L2-normalize
        # rows so cosine similarity is a plain dot product.
        tf = X / np.maximum(X.sum(1, keepdims=True), 1e-9)
        M = tf * self.idf[None, :]
        norm = np.sqrt((M ** 2).sum(1, keepdims=True))
        return M / np.maximum(norm, 1e-9)

    def fit(self, docs, intents, responses):
        self.intents = np.asarray(intents)
        self.responses = responses               # intent -> reply string
        # Vocabulary = every word seen in training utterances.
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        X = self._counts(docs)
        # idf = log(N / df) + 1 ; rarer words carry more discriminative weight.
        df = (X > 0).sum(0)
        self.idf = np.log(X.shape[0] / np.maximum(df, 1)) + 1.0
        self.M = self._tfidf(X)                  # normalized training matrix
        return self

    def predict(self, docs):
        # Cosine similarity to every training utterance; nearest one's intent.
        Q = self._tfidf(self._counts(docs))
        sims = Q @ self.M.T
        return self.intents[np.argmax(sims, axis=1)]

    def respond(self, text):
        return self.responses[self.predict([text])[0]]


def make_dialogues(seed=0):
    # Synthetic support corpus: each intent owns a phrase bank; utterances mix
    # intent phrases with shared filler so classes overlap yet stay separable.
    rng = np.random.RandomState(seed)
    banks = {
        "greeting":  ["hi", "hello", "hey there", "good morning", "greetings"],
        "goodbye":   ["bye", "goodbye", "see you later", "talk soon", "farewell"],
        "order":     ["where is my order", "track my package", "delivery status",
                      "when will it arrive", "my shipment location"],
        "refund":    ["i want a refund", "return this item", "get my money back",
                      "cancel my order", "how to return a product"],
        "hours":     ["what time do you open", "business hours", "when do you close",
                      "opening times", "are you open now"],
    }
    replies = {
        "greeting": "Hello! How can I help you today?",
        "goodbye":  "Goodbye! Have a great day.",
        "order":    "You can track your order in the Orders tab.",
        "refund":   "Sure, I can start a refund for you.",
        "hours":    "We are open 9am to 6pm, Monday to Friday.",
    }
    filler = ["please", "hey", "so", "um", "could you", "thanks", "just"]

    docs, intents = [], []
    for intent, phrases in banks.items():
        for _ in range(60):                      # 60 utterances per intent
            base = rng.choice(phrases)
            words = tokenize(base)
            if rng.rand() < 0.6:                 # sprinkle filler on 60% of turns
                pos = rng.randint(0, len(words) + 1)
                words = words[:pos] + [rng.choice(filler)] + words[pos:]
            docs.append(" ".join(words))
            intents.append(intent)
    return docs, np.array(intents), replies


if __name__ == "__main__":
    np.random.seed(0)

    docs, intents, replies = make_dialogues(seed=0)
    # Held-out split: fit retrieval index on 70%, evaluate on unseen 30%.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Xtr = [docs[i] for i in tr]
    Xte = [docs[i] for i in te]
    ytr, yte = intents[tr], intents[te]

    bot = ChatBot().fit(Xtr, ytr, replies)
    pred = bot.predict(Xte)
    acc = np.mean(pred == yte)

    # Majority-class baseline: always guess the most common training intent.
    vals, cnts = np.unique(ytr, return_counts=True)
    majority = vals[np.argmax(cnts)]
    base_acc = np.mean(yte == majority)

    print("Utterances: %d   Train: %d   Test: %d   Intents: %d"
          % (len(docs), len(tr), len(te), len(replies)))
    print("Vocabulary size: %d" % len(bot.vocab))
    print("-" * 56)
    print("Chatbot   intent accuracy: %.4f" % acc)
    print("Majority  intent accuracy: %.4f  (always '%s')" % (base_acc, majority))
    print("-" * 56)
    for q in ["hello there please", "where is my package",
              "i want my money back", "what time do you open"]:
        print("  user: %-26s bot: %s" % (q, bot.respond(q)))
    print("-" * 56)
    print("Chatbot beats majority baseline: %s" % (acc > base_acc))
