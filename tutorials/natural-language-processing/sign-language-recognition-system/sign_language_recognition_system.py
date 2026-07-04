import numpy as np

# Sign-language recognition from hand-landmark features.
# Each sign (a fingerspelled letter) is a distinct "hand pose": a vector of
# finger-bend / orientation descriptors. Samples are a class prototype plus
# sensor noise, so the latent per-letter structure is recoverable. A from-
# scratch softmax classifier learns to map descriptors -> letters, and we then
# decode a noisy stream of signs back into the spelled word (the NLP payoff).

LETTERS = list("HELOWRD")          # enough to fingerspell "HELLO WORLD"


class SoftmaxClassifier:
    """Multinomial logistic regression trained by manual batch gradient descent."""

    def __init__(self, epochs=300, lr=0.5, reg=1e-3, seed=0):
        self.epochs = epochs
        self.lr = lr
        self.reg = reg            # L2 weight decay
        self.seed = seed

    def _standardize(self, X, fit=False):
        # Zero-mean / unit-variance per feature -> stable, fast gradient descent.
        if fit:
            self.mu = X.mean(0)
            self.sd = X.std(0) + 1e-8
        return (X - self.mu) / self.sd

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(1, keepdims=True)          # numerical stability
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        Xs = self._standardize(np.asarray(X, float), fit=True)
        n, d = Xs.shape
        Xb = np.hstack([Xs, np.ones((n, 1))])    # bias column
        self.K = int(y.max()) + 1
        Y = np.eye(self.K)[y]                     # one-hot targets
        self.W = rng.randn(d + 1, self.K) * 0.01

        for _ in range(self.epochs):
            P = self._softmax(Xb @ self.W)
            grad = Xb.T @ (P - Y) / n + self.reg * self.W
            self.W -= self.lr * grad
        return self

    def predict_proba(self, X):
        Xs = self._standardize(np.asarray(X, float))
        Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
        return self._softmax(Xb @ self.W)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def make_signs(n_per=90, dim=16, noise=1.4, seed=0):
    # Plant one random hand-pose prototype per letter, then draw noisy samples.
    rng = np.random.RandomState(seed)
    K = len(LETTERS)
    prototypes = rng.uniform(-2.5, 2.5, size=(K, dim))   # distinct poses
    X, y = [], []
    for c in range(K):
        X.append(prototypes[c] + rng.randn(n_per, dim) * noise)
        y.append(np.full(n_per, c))
    X = np.vstack(X)
    y = np.concatenate(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], prototypes


def spell(word, prototypes, noise=1.4, seed=1):
    # Turn a word into a stream of noisy sign samples (one per letter).
    rng = np.random.RandomState(seed)
    idx = [LETTERS.index(ch) for ch in word]
    samples = np.array([prototypes[i] + rng.randn(prototypes.shape[1]) * noise
                        for i in idx])
    return samples, np.array(idx)


if __name__ == "__main__":
    np.random.seed(0)

    X, y, protos = make_signs(n_per=90, dim=16, noise=1.4, seed=0)

    # Held-out split: train on 70%, test on the rest.
    split = int(0.7 * len(y))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    clf = SoftmaxClassifier(epochs=300, lr=0.5, reg=1e-3).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = np.mean(pred == yte)

    # Majority-class baseline: always guess the most frequent training letter.
    majority = np.bincount(ytr).argmax()
    base_acc = np.mean(yte == majority)

    print("Signs: %d samples, %d letters (%s), %d features"
          % (len(y), len(LETTERS), "".join(LETTERS), X.shape[1]))
    print("Train: %d   Test: %d" % (len(ytr), len(yte)))
    print("-" * 56)
    print("Softmax recognizer accuracy: %.4f" % acc)
    print("Majority baseline accuracy:  %.4f  (chance = %.4f)"
          % (base_acc, 1.0 / len(LETTERS)))
    print("-" * 56)

    # NLP payoff: decode a noisy stream of signs back into a word.
    for word in ["HELLO", "WORLD"]:
        samples, true_idx = spell(word, protos, noise=1.4, seed=len(word))
        decoded = "".join(LETTERS[i] for i in clf.predict(samples))
        char_acc = np.mean([d == t for d, t in zip(decoded, word)])
        print("  spelled '%s' -> recognized '%s'   char-acc %.2f"
              % (word, decoded, char_acc))
    print("-" * 56)
    print("Beats majority baseline: %s" % (acc > base_acc))
