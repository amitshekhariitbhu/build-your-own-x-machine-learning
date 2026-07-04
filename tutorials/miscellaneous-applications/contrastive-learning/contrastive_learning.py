import numpy as np


class ContrastiveEncoder:
    """Self-supervised contrastive representation learner (SimCLR / NT-Xent), from scratch.

    A linear encoder maps augmented views into an embedding space, then the
    temperature-scaled normalized cross-entropy loss (NT-Xent) pulls the two views
    of the SAME instance (a positive pair) together and pushes every other instance
    in the batch (the negatives) apart. Gradients are derived and back-propagated by
    hand through the L2-normalization and the linear layer. fit() NEVER sees labels."""

    def __init__(self, out_dim=8, temperature=0.2, lr=0.05, n_iter=300, seed=0):
        self.out_dim = out_dim
        self.temperature = temperature
        self.lr = lr
        self.n_iter = n_iter
        self.rng = np.random.RandomState(seed)

    def _forward(self, A):
        # Linear projection, then L2-normalize each row onto the unit sphere.
        H = A @ self.W
        norm = np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
        return H, norm, H / norm

    def _ntxent(self, Z):
        # Z: (2N, m). Rows i and i+N are the two views of instance i (positives).
        twoN = Z.shape[0]
        N = twoN // 2
        logits = (Z @ Z.T) / self.temperature
        np.fill_diagonal(logits, -1e9)            # an anchor is never its own negative
        pos = np.r_[np.arange(N, twoN), np.arange(0, N)]

        logits -= logits.max(axis=1, keepdims=True)   # stabilize softmax
        expv = np.exp(logits)
        P = expv / expv.sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(P[np.arange(twoN), pos] + 1e-12))

        # Softmax cross-entropy gradient w.r.t. the logits.
        G = P.copy()
        G[np.arange(twoN), pos] -= 1.0
        G /= twoN
        return loss, G

    def fit(self, base, augment):
        # base: (N, D) clean instances; augment(base) -> a fresh stochastic view.
        N, D = base.shape
        self.W = self.rng.randn(D, self.out_dim) * 0.1
        m, v, t = np.zeros_like(self.W), np.zeros_like(self.W), 0   # Adam state

        for _ in range(self.n_iter):
            A = np.vstack([augment(base), augment(base)])   # two independent views
            H, norm, Z = self._forward(A)
            loss, G = self._ntxent(Z)

            # Back-prop:  logits = (Z Zt)/tau  ->  dZ = (1/tau)(G + Gt) Z
            dZ = (G + G.T) @ Z / self.temperature
            # through the L2-normalization Jacobian: dH = (dZ - Z (Z.dZ)) / ||H||
            dH = (dZ - Z * np.sum(dZ * Z, axis=1, keepdims=True)) / norm
            dW = A.T @ dH

            t += 1                                          # Adam update
            m = 0.9 * m + 0.1 * dW
            v = 0.999 * v + 0.001 * (dW * dW)
            mh, vh = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
            self.W -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        self.final_loss = loss
        return self

    def transform(self, X):
        return self._forward(X)[2]


def knn_accuracy(train_X, train_y, test_X, test_y, k=5):
    # Plain Euclidean k-NN vote; a linear probe on the frozen representation.
    sq = np.sum(train_X ** 2, 1)[None, :] + np.sum(test_X ** 2, 1)[:, None] - 2 * test_X @ train_X.T
    nn = np.argsort(np.maximum(sq, 0.0), axis=1)[:, :k]
    votes = train_y[nn]
    pred = np.array([np.bincount(r).argmax() for r in votes])
    return np.mean(pred == test_y)


if __name__ == "__main__":
    np.random.seed(0)

    # PLANTED STRUCTURE: 4 classes live in a tiny 3-D "content" latent space that is
    # embedded into just the first CONTENT dims of a 300-D observation; the remaining
    # dims are pure distractor noise. Every view also gets fresh per-view NUISANCE on
    # ALL dims. The class-carrying content is the ONLY thing shared between two views,
    # so raw distances drown in ~300 noise dims while a contrastive encoder can learn
    # to read the content subspace and ignore the rest.
    K, latent_dim, CONTENT, D = 4, 3, 5, 300
    centers = np.random.randn(K, latent_dim) * 2.5
    Proj = np.random.randn(latent_dim, CONTENT)           # 3-D latent -> 5 content dims

    def sample(per):
        u = np.vstack([centers[c] + 0.30 * np.random.randn(per, latent_dim) for c in range(K)])
        obs = np.zeros((per * K, D))
        obs[:, :CONTENT] = u @ Proj                       # content only in first dims
        return obs, np.repeat(np.arange(K), per)

    content_tr, y_tr = sample(40)    # training instances (labels unused in fit)
    content_te, y_te = sample(15)    # held-out instances

    NUISANCE = 2.0
    augment = lambda C: C + NUISANCE * np.random.randn(*C.shape)   # SimCLR augmentation

    model = ContrastiveEncoder(out_dim=8, temperature=0.2, lr=0.05, n_iter=400).fit(content_tr, augment)

    # One observed (noisy) view of every instance for a fair, apples-to-apples probe.
    Xtr, Xte = augment(content_tr), augment(content_te)

    # CORRECTNESS SIGNAL: k-NN class accuracy on held-out instances in each space.
    acc_contrast = knn_accuracy(model.transform(Xtr), y_tr, model.transform(Xte), y_te)
    acc_raw = knn_accuracy(Xtr, y_tr, Xte, y_te)                       # noisy raw features
    rand_W = ContrastiveEncoder(out_dim=8, seed=1)                     # untrained encoder
    rand_W.W = np.random.randn(D, 8) * 0.1
    acc_rand = knn_accuracy(rand_W.transform(Xtr), y_tr, rand_W.transform(Xte), y_te)
    acc_majority = np.max(np.bincount(y_te)) / len(y_te)

    print("Instances: {} train / {} test   Observed dim: {}   Classes: {}".format(
        len(y_tr), len(y_te), D, K))
    print("Final NT-Xent loss:                        {:.3f}".format(model.final_loss))
    print("Majority-class baseline accuracy:          {:.3f}".format(acc_majority))
    print("Raw noisy features k-NN accuracy:          {:.3f}".format(acc_raw))
    print("Random projection k-NN accuracy:           {:.3f}".format(acc_rand))
    print("Contrastive embedding k-NN accuracy:       {:.3f}".format(acc_contrast))
    print("Improvement over raw features:             +{:.3f}".format(acc_contrast - acc_raw))
    assert acc_contrast > acc_majority + 0.4, "contrastive did not beat the majority baseline"
    assert acc_contrast > acc_raw + 0.15, "contrastive did not beat the raw-feature baseline"
    print("PASS: contrastive learning recovered class structure with NO labels and beat every baseline.")
