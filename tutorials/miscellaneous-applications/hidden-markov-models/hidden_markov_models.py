import numpy as np

EPS = 1e-12


class HiddenMarkovModel:
    """Discrete-emission HMM trained unsupervised with Baum-Welch (EM).

    Parameters:
      pi : initial state distribution        (N,)
      A  : state transition matrix A[i,j]     (N, N)  P(next=j | cur=i)
      B  : emission matrix B[j,k]             (N, M)  P(obs=k | state=j)
    """

    def __init__(self, n_states, n_obs):
        self.N, self.M = n_states, n_obs
        self.pi = self.A = self.B = None

    # ---- scaled forward-backward (Rabiner scaling avoids underflow) ----
    def _forward(self, o):
        T = len(o)
        alpha = np.zeros((T, self.N))
        c = np.zeros(T)                       # per-step scaling factors
        alpha[0] = self.pi * self.B[:, o[0]]
        c[0] = alpha[0].sum() + EPS
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, o[t]]
            c[t] = alpha[t].sum() + EPS
            alpha[t] /= c[t]
        return alpha, c                       # log-likelihood = sum(log c)

    def _backward(self, o, c):
        T = len(o)
        beta = np.zeros((T, self.N))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, o[t + 1]] * beta[t + 1]) / c[t + 1]
        return beta

    def loglik(self, o):
        _, c = self._forward(o)
        return np.log(c).sum()

    def fit(self, seqs, n_iter=40, seed=0):
        """Baum-Welch over a list of observation sequences. Returns LL history."""
        rng = np.random.RandomState(seed)
        # Random but valid (row-stochastic) initialization.
        self.pi = rng.dirichlet(np.ones(self.N))
        self.A = rng.dirichlet(np.ones(self.N), size=self.N)
        self.B = rng.dirichlet(np.ones(self.M), size=self.N)

        history = []
        for _ in range(n_iter):
            pi_acc = np.zeros(self.N)
            A_num = np.zeros((self.N, self.N))
            A_den = np.zeros(self.N)
            B_num = np.zeros((self.N, self.M))
            B_den = np.zeros(self.N)
            total_ll = 0.0

            for o in seqs:
                T = len(o)
                alpha, c = self._forward(o)
                beta = self._backward(o, c)
                total_ll += np.log(c).sum()

                # gamma[t,i] = P(state_t=i | obs); normalize per step.
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + EPS

                # xi[t,i,j] = P(state_t=i, state_{t+1}=j | obs).
                xi = (alpha[:-1, :, None] * self.A[None] *
                      self.B[:, o[1:]].T[:, None, :] * beta[1:, None, :])
                xi /= xi.sum(axis=(1, 2), keepdims=True) + EPS

                pi_acc += gamma[0]
                A_num += xi.sum(axis=0)
                A_den += gamma[:-1].sum(axis=0)
                for k in range(self.M):
                    B_num[:, k] += gamma[o == k].sum(axis=0)
                B_den += gamma.sum(axis=0)

            # M-step: renormalize the expected counts.
            self.pi = pi_acc / len(seqs)
            self.A = A_num / (A_den[:, None] + EPS)
            self.B = B_num / (B_den[:, None] + EPS)
            history.append(total_ll)
        return history

    def viterbi(self, o):
        """Most likely hidden-state path (max-product in log space)."""
        T = len(o)
        lpi, lA, lB = np.log(self.pi + EPS), np.log(self.A + EPS), np.log(self.B + EPS)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        delta[0] = lpi + lB[:, o[0]]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + lA        # (i -> j)
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores.max(axis=0) + lB[:, o[t]]
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path


def sample_hmm(pi, A, B, T):
    """Draw one (states, observations) pair from a known HMM."""
    N, M = A.shape[0], B.shape[1]
    states, obs = np.zeros(T, int), np.zeros(T, int)
    states[0] = np.random.choice(N, p=pi)
    obs[0] = np.random.choice(M, p=B[states[0]])
    for t in range(1, T):
        states[t] = np.random.choice(N, p=A[states[t - 1]])
        obs[t] = np.random.choice(M, p=B[states[t]])
    return states, obs


if __name__ == "__main__":
    np.random.seed(0)

    # --- True HMM: 2 sticky hidden states with distinct emission profiles ---
    pi_t = np.array([0.6, 0.4])
    A_t = np.array([[0.90, 0.10],
                    [0.10, 0.90]])          # states persist, so runs are long
    B_t = np.array([[0.70, 0.20, 0.10],     # state 0 favors symbol 0
                    [0.10, 0.20, 0.70]])    # state 1 favors symbol 2

    train = [sample_hmm(pi_t, A_t, B_t, 100)[1] for _ in range(30)]
    test = [sample_hmm(pi_t, A_t, B_t, 100) for _ in range(10)]

    hmm = HiddenMarkovModel(n_states=2, n_obs=3)
    hist = hmm.fit(train, n_iter=40)

    # 1) EM must monotonically increase data log-likelihood.
    monotone = all(b >= a - 1e-6 for a, b in zip(hist, hist[1:]))

    # 2) Viterbi should recover hidden states on held-out sequences.
    #    Labels are permutation-invariant (2 states), so score both alignments.
    correct = total = 0
    for s_true, o in test:
        pred = hmm.viterbi(o)
        acc = max((pred == s_true).mean(), (pred != s_true).mean())
        correct += acc * len(o)
        total += len(o)
    viterbi_acc = correct / total

    # Baseline: always predict the majority state (best constant guess).
    all_true = np.concatenate([s for s, _ in test])
    majority = max(np.mean(all_true == 0), np.mean(all_true == 1))

    print("Baum-Welch log-likelihood: {:.1f} -> {:.1f}".format(hist[0], hist[-1]))
    print("Log-likelihood monotonically increasing:", monotone)
    print("Learned transition matrix A:\n", np.round(hmm.A, 2))
    print("Learned emission matrix  B:\n", np.round(hmm.B, 2))
    print("-" * 40)
    print("Majority-state baseline accuracy: {:.3f}".format(majority))
    print("Viterbi state-recovery accuracy:  {:.3f}".format(viterbi_acc))
    print("Beats baseline:", viterbi_acc > majority + 0.05)
