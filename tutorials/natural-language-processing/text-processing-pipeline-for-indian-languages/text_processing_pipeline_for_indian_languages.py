import numpy as np

# Text-processing pipeline for Indian-language (Devanagari) text, from scratch.
# Real Indic text needs steps that ASCII NLP skips: stripping invisible ZWJ/ZWNJ
# join controls, folding the combining nukta, splitting on the danda '।', dropping
# grammatical stopwords, and light suffix stemming to collapse inflected forms.
# We plant exactly this noise into synthetic Hindi topic text, then show the
# pipeline + a from-scratch Multinomial Naive Bayes recovers the topic far better
# than raw whitespace tokens, and both beat the majority-class baseline.

ZWJ, ZWNJ, ZWSP, NUKTA, BOM = "‍", "‌", "​", "़", "﻿"
DANDA, DDANDA = "।", "॥"                      # '।' and '॥' sentence marks

STOPWORDS = {"है", "और", "के", "में", "को", "का", "की", "यह",
             "से", "पर", "ने", "भी", "था", "हैं", "एक"}

# Inflectional suffixes stripped by the light stemmer (longest first) so that
# surface variants like खेल / खेलों / खेलें collapse to one stem.
SUFFIXES = ["ियों", "ाओं", "ुओं", "ों", "ें", "ीं", "ने", "ना",
            "नी", "ता", "ती", "ते", "ा", "ी", "े", "ो", "ू", "ँ"]


class IndicPipeline:
    """Normalize -> tokenize -> drop stopwords -> light-stem Devanagari text."""

    def normalize(self, text):
        # Delete zero-width join controls and the combining nukta; these are the
        # invisible encoding variants that otherwise fragment a token's identity.
        drop = {ZWJ, ZWNJ, ZWSP, NUKTA, BOM}
        return "".join(ch for ch in text if ch not in drop)

    def tokenize(self, text):
        # Danda/double-danda are the Devanagari full stops; then split on spaces.
        for d in (DANDA, DDANDA, "|", ".", ","):
            text = text.replace(d, " ")
        return [t for t in text.split() if t]

    def stem(self, token):
        # Strip one longest matching suffix, keeping at least 2 code points.
        for suf in SUFFIXES:
            if token.endswith(suf) and len(token) - len(suf) >= 2:
                return token[: -len(suf)]
        return token

    def process(self, text, stopwords=True, stemming=True):
        toks = self.tokenize(self.normalize(text))
        if stopwords:
            toks = [t for t in toks if t not in STOPWORDS]
        if stemming:
            toks = [self.stem(t) for t in toks]
        return toks


class MultinomialNaiveBayes:
    """Bag-of-words classifier: count word frequencies per class, score in logs."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha                                 # Laplace smoothing

    def _vectorize(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, toks in enumerate(docs):
            for w in toks:
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] += 1.0
        return X

    def fit(self, docs, y):
        y = np.asarray(y)
        self.vocab = {}
        for toks in docs:
            for w in toks:
                self.vocab.setdefault(w, len(self.vocab))
        X = self._vectorize(docs)
        self.classes = np.unique(y)
        self.log_prior = np.zeros(len(self.classes))
        self.log_lik = np.zeros((len(self.classes), len(self.vocab)))
        for k, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[k] = np.log(Xc.shape[0] / X.shape[0])
            counts = Xc.sum(0) + self.alpha               # P(word|class), smoothed
            self.log_lik[k] = np.log(counts / counts.sum())
        return self

    def predict(self, docs):
        X = self._vectorize(docs)
        scores = self.log_prior[None, :] + X @ self.log_lik.T
        return self.classes[np.argmax(scores, axis=1)]


TOPICS = {                                                 # planted topic vocab
    0: ["खेल", "गेंद", "बल्ला", "विकेट", "मैच", "पारी", "चौका", "रन", "कप्तान"],
    1: ["रोटी", "चावल", "सब्जी", "दाल", "मसाला", "स्वाद", "मिठाई", "भोजन", "अचार"],
    2: ["बारिश", "धूप", "बादल", "ठंड", "गर्मी", "हवा", "मौसम", "कोहरा", "सर्दी"],
}
INFLECT = ["", "", "", "ों", "ें", "े", "ी", "ने"]         # planted inflections


def _corrupt(word, rng):
    # Add invisible join controls / a stray nukta that normalization must undo.
    if rng.rand() < 0.4:
        p = rng.randint(1, len(word) + 1)
        word = word[:p] + rng.choice([ZWJ, ZWNJ]) + word[p:]
    if rng.rand() < 0.3:
        p = rng.randint(1, len(word) + 1)
        word = word[:p] + NUKTA + word[p:]
    return word


def make_corpus(n=480, seed=0):
    # Each sentence samples content words from one topic, inflects and corrupts
    # them, and sprinkles in shared stopwords + a danda -> recoverable structure.
    rng = np.random.RandomState(seed)
    stop = list(STOPWORDS)
    docs, labels = [], []
    for _ in range(n):
        topic = rng.randint(3)
        k = rng.randint(4, 8)
        toks = []
        for w in rng.choice(TOPICS[topic], k):
            toks.append(_corrupt(w + rng.choice(INFLECT), rng))
            if rng.rand() < 0.5:
                toks.append(rng.choice(stop))
        rng.shuffle(toks)
        docs.append(" ".join(toks) + " " + DANDA)
        labels.append(topic)
    return docs, np.array(labels)


if __name__ == "__main__":
    np.random.seed(0)
    docs, y = make_corpus(n=480, seed=0)

    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    raw_tr, raw_te = [docs[i] for i in tr], [docs[i] for i in te]
    ytr, yte = y[tr], y[te]

    pipe = IndicPipeline()
    # Full pipeline: normalize + stopword removal + stemming.
    Ptr = [pipe.process(d) for d in raw_tr]
    Pte = [pipe.process(d) for d in raw_te]
    # Raw baseline: whitespace/danda split only, no cleaning (ZWJ, nukta, suffixes).
    Rtr = [pipe.tokenize(d) for d in raw_tr]
    Rte = [pipe.tokenize(d) for d in raw_te]

    pipe_acc = np.mean(MultinomialNaiveBayes().fit(Ptr, ytr).predict(Pte) == yte)
    raw_acc = np.mean(MultinomialNaiveBayes().fit(Rtr, ytr).predict(Rte) == yte)
    majority = np.bincount(ytr).argmax()
    base_acc = np.mean(yte == majority)

    demo = "गेंद और विकेट पर‍ बल्ले़बाज ने चौके ।"
    print("Docs: %d   Train: %d   Test: %d   Classes: 3" % (len(docs), len(tr), len(te)))
    print("Vocabulary  raw (no pipeline): %d   after pipeline: %d"
          % (len({w for d in Rtr for w in d}), len({w for d in Ptr for w in d})))
    print("Sample raw tokens :", pipe.tokenize(demo))
    print("Sample processed  :", pipe.process(demo))
    print("-" * 60)
    print("Naive Bayes + pipeline  accuracy: %.4f" % pipe_acc)
    print("Naive Bayes + raw text  accuracy: %.4f" % raw_acc)
    print("Majority-class baseline accuracy: %.4f" % base_acc)
    print("-" * 60)
    ok = pipe_acc > raw_acc and pipe_acc > base_acc + 0.3
    print("Pipeline beats raw and baseline: %s" % ok)
    print("SUCCESS" if ok else "FAIL")
