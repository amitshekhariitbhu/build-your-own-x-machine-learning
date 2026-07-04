import numpy as np


class TSNE:
    """t-Distributed Stochastic Neighbor Embedding, from scratch.

    High-dim neighbor probabilities P (Gaussian, perplexity-calibrated per point)
    are matched to low-dim probabilities Q (heavy-tailed Student-t) by minimizing
    KL(P||Q) with momentum gradient descent and early exaggeration."""

    def __init__(self, n_components=2, perplexity=20.0, learning_rate=200.0,
                 n_iter=400, early_exaggeration=4.0, exaggerate_iter=100):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.exaggerate_iter = exaggerate_iter

    @staticmethod
    def _pairwise_sq_dists(X):
        # ||xi - xj||^2 = |xi|^2 + |xj|^2 - 2 xi.xj, computed vectorized.
        sq = np.sum(X * X, axis=1)
        D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        return np.maximum(D, 0.0)

    def _p_conditional(self, D):
        # For each point, binary-search beta=1/(2 sigma^2) so the row's perplexity
        # (2^entropy) matches the target. Returns the conditional P_{j|i}.
        n = D.shape[0]
        P = np.zeros((n, n))
        target = np.log(self.perplexity)
        for i in range(n):
            betamin, betamax = -np.inf, np.inf
            beta = 1.0
            Di = np.delete(D[i], i)  # distances to all other points
            for _ in range(50):
                Pi = np.exp(-Di * beta)
                s = Pi.sum()
                if s == 0.0:
                    s = 1e-12
                Pi /= s
                # Shannon entropy H = log(sum) + beta*<D> of the neighbor dist.
                H = np.log(s) + beta * np.sum(Di * Pi)
                if H > target:  # perplexity too high -> increase beta
                    betamin = beta
                    beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
                else:
                    betamax = beta
                    beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2
                if abs(H - target) < 1e-5:
                    break
            P[i, np.arange(n) != i] = Pi
        return P

    def fit_transform(self, X):
        n = X.shape[0]
        D = self._pairwise_sq_dists(X)

        # Symmetric joint distribution P over pairs.
        P = self._p_conditional(D)
        P = (P + P.T) / (2.0 * n)
        P = np.maximum(P, 1e-12)
        P *= self.early_exaggeration  # early exaggeration sharpens cluster gaps

        # Small random init; momentum buffer.
        Y = np.random.randn(n, self.n_components) * 1e-4
        vel = np.zeros_like(Y)

        for it in range(self.n_iter):
            if it == self.exaggerate_iter:
                P /= self.early_exaggeration  # release exaggeration

            # Low-dim heavy-tailed affinities Q.
            Dy = self._pairwise_sq_dists(Y)
            num = 1.0 / (1.0 + Dy)
            np.fill_diagonal(num, 0.0)
            Q = np.maximum(num / num.sum(), 1e-12)

            # Gradient of KL(P||Q): 4 * sum_j (P-Q)*num*(yi-yj).
            PQ = (P - Q) * num
            grad = 4.0 * ((np.diag(PQ.sum(axis=1)) - PQ) @ Y)

            momentum = 0.5 if it < 20 else 0.8
            vel = momentum * vel - self.learning_rate * grad
            Y = Y + vel
            Y = Y - Y.mean(axis=0)  # keep embedding centered

        return Y


def nn_purity(Y, labels):
    # Fraction of points whose nearest neighbor shares their label: a direct
    # measure of how well local class structure survived in the embedding.
    D = TSNE._pairwise_sq_dists(Y)
    np.fill_diagonal(D, np.inf)
    nn = np.argmin(D, axis=1)
    return np.mean(labels[nn] == labels)


if __name__ == "__main__":
    np.random.seed(0)

    # PLANTED STRUCTURE: 4 well-separated Gaussian blobs living in a 30-dim space.
    # The blobs are the latent classes t-SNE must pull apart in 2-D.
    n_per, dim, k = 40, 30, 4
    centers = np.random.randn(k, dim) * 9.0
    X = np.vstack([c + np.random.randn(n_per, dim) for c in centers])
    labels = np.repeat(np.arange(k), n_per)

    Y = TSNE(perplexity=20.0, n_iter=400).fit_transform(X)

    # CORRECTNESS SIGNAL: nearest-neighbor label purity in the 2-D embedding.
    tsne_purity = nn_purity(Y, labels)

    # BASELINE 1: a random 2-D layout ignores structure entirely.
    rand_purity = nn_purity(np.random.randn(len(X), 2), labels)
    # BASELINE 2: chance level for a random neighbor given the class sizes.
    chance = np.sum((np.bincount(labels) / len(labels)) ** 2)

    print("Points: {}  Input dim: {}  Planted clusters: {}".format(len(X), dim, k))
    print("Chance NN-purity (random neighbor):   {:.3f}".format(chance))
    print("Random 2-D layout NN-purity:          {:.3f}".format(rand_purity))
    print("t-SNE embedding NN-purity:            {:.3f}".format(tsne_purity))
    print("Improvement over random layout:       +{:.3f}".format(tsne_purity - rand_purity))
    assert tsne_purity > 0.95, "t-SNE failed to preserve cluster structure"
    assert tsne_purity > rand_purity + 0.3, "t-SNE did not beat the random baseline"
    print("PASS: t-SNE recovered the planted clusters and crushed both baselines.")
