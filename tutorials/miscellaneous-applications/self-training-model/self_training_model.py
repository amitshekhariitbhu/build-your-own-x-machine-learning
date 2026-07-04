import numpy as np

# Self-training: a semi-supervised wrapper, from scratch.
# We hold a TINY labeled set and a LARGE unlabeled pool drawn from the same clusters.
# Self-training bootstraps the labels: fit the base model, ask it for class PROBABILITIES
# on the unlabeled pool, and promote only the MOST-confident predictions (proba >= tau)
# to "pseudo-labels". Refit on labeled + pseudo-labeled and repeat. The base learner is a
# hand-written Parzen-window (RBF-kernel) classifier whose confidence is LOCAL -- it decays
# away from labeled points -- so each round only the ring of pool points hugging the current
# labels is trusted. Those become labels, extending the field so the next ring can be
# absorbed. On two interleaving moons the labels flow ALONG each curved manifold and stop
# at the low-density gap, recovering a boundary a few labels alone could never place.
# Everything below (kernel scores, confidence gating, iterative refit) is manual numpy.


class ParzenClassifier:
    """Kernel-density classifier: class score = sum of RBF kernels to that class's rows."""

    def __init__(self, bw=0.2):
        self.bw = bw                                    # bandwidth => how far a label's vote reaches

    def fit(self, X, y):
        self.X, self.y = X, y
        self.k = int(y.max()) + 1
        return self

    def predict_proba(self, X):
        d2 = ((X[:, None, :] - self.X[None, :, :]) ** 2).sum(2)   # (m, n_train) squared dists
        K = np.exp(-d2 / (2 * self.bw ** 2))                      # local RBF weights
        S = np.stack([K[:, self.y == c].sum(1) for c in range(self.k)], axis=1)
        S = S + 1e-12
        return S / S.sum(1, keepdims=True)              # normalize kernel mass -> class proba

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


class SelfTrainingClassifier:
    """Wrap any proba base learner; iteratively absorb its high-confidence pseudo-labels."""

    def __init__(self, base_factory, tau=0.99, max_rounds=30):
        self.base_factory = base_factory                # () -> a fresh base model
        self.tau, self.max_rounds = tau, max_rounds

    def fit(self, X_lab, y_lab, X_unlab):
        X, y = X_lab.copy(), y_lab.copy()
        pool = X_unlab.copy()                           # unlabeled rows still up for grabs
        self.n_pseudo, self.rounds = 0, 0
        for _ in range(self.max_rounds):
            self.model = self.base_factory().fit(X, y)
            self.rounds += 1
            if len(pool) == 0:
                break
            proba = self.model.predict_proba(pool)
            take = proba.max(1) >= self.tau             # confident enough to trust this round
            if not take.any():
                break                                   # nothing new absorbed -> converged
            X = np.vstack([X, pool[take]])              # promote to pseudo-labeled
            y = np.concatenate([y, proba[take].argmax(1)])
            pool = pool[~take]
            self.n_pseudo += int(take.sum())
        self.model = self.base_factory().fit(X, y)      # final fit on everything gathered
        return self

    def predict(self, X):
        return self.model.predict(X)


if __name__ == "__main__":
    np.random.seed(0)

    # PLANTED STRUCTURE: two interleaving half-moons (a nonlinear, cluster-structured
    # boundary). Only 3 labels PER CLASS are revealed (6 labeled rows) alongside a large
    # UNLABELED pool and a held-out test set from the same moons. Six points cannot trace
    # each curved arm, but the moons are dense and separated by a low-density gap -- exactly
    # the assumption self-training exploits: confident pseudo-labels flow along each arm and
    # halt at the gap, so the recovered boundary beats a model trained on the 6 labels alone.
    def make_moons(n_per, noise):
        t = np.linspace(0, np.pi, n_per)
        arm0 = np.c_[np.cos(t), np.sin(t)]                       # upper arc  -> class 0
        arm1 = np.c_[1 - np.cos(t), 1 - np.sin(t) - 0.5]         # lower arc  -> class 1
        X = np.vstack([arm0, arm1]) + np.random.randn(2 * n_per, 2) * noise
        y = np.r_[np.zeros(n_per), np.ones(n_per)].astype(int)
        idx = np.random.permutation(len(y))
        return X[idx], y[idx]

    NOISE = 0.12
    X_lab, y_lab = make_moons(3, NOISE)                 # 6 labeled rows total
    X_unlab, _ = make_moons(150, NOISE)                 # 300-row unlabeled pool
    X_te, y_te = make_moons(200, NOISE)                 # 400-row held-out test set

    factory = lambda: ParzenClassifier(bw=0.2)
    majority = max(np.bincount(y_te)) / len(y_te)       # predict-the-majority baseline

    sup = factory().fit(X_lab, y_lab)                   # supervised on the 6 labels only
    sup_acc = float(np.mean(sup.predict(X_te) == y_te))

    st = SelfTrainingClassifier(factory, tau=0.99, max_rounds=30).fit(X_lab, y_lab, X_unlab)
    st_acc = float(np.mean(st.predict(X_te) == y_te))

    print("Task: two interleaving moons; 6 labeled + 300 unlabeled rows (semi-supervised)")
    print("-" * 64)
    print("Majority-class baseline acc      : {:.3f}".format(majority))
    print("Supervised (6 labels only) acc   : {:.3f}".format(sup_acc))
    print("Self-training            acc     : {:.3f}".format(st_acc))
    print("Pseudo-labels absorbed           : {} of {} pool rows in {} rounds".format(
        st.n_pseudo, len(X_unlab), st.rounds))
    print("Self-training lift over supervised: {:+.3f}".format(st_acc - sup_acc))

    assert st_acc > majority + 0.15, "self-training must clearly beat the majority baseline"
    assert st_acc > sup_acc + 0.05, "self-training must clearly beat supervised-on-labels-only"
    print("PASS: self-training beat the majority baseline AND supervised-only by a clear margin.")
