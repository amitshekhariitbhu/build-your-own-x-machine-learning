import numpy as np


def newton_minimize(grad, hess, x0, n_iter=20, tol=1e-10):
    """Generic Newton's method: x <- x - H(x)^-1 g(x) until the step is tiny.

    grad(x) -> gradient vector, hess(x) -> Hessian matrix. Returns (x, n_steps).
    """
    x = np.asarray(x0, dtype=float).copy()
    for step in range(n_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:      # already at a stationary point
            return x, step
        delta = np.linalg.solve(hess(x), g)  # solve H delta = g (no explicit inverse)
        x = x - delta
    return x, n_iter


class NewtonLogisticRegression:
    """Logistic regression trained by Newton-Raphson (a.k.a. IRLS).

    Minimizes the negative log-likelihood; each step uses the exact gradient
    and Hessian, so it converges in a handful of iterations.
    """

    def __init__(self, n_iter=15, ridge=1e-6):
        self.n_iter = n_iter
        self.ridge = ridge  # tiny L2 to keep the Hessian well-conditioned
        self.w = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    @staticmethod
    def _add_bias(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        Xb = self._add_bias(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        n, d = Xb.shape
        self.w = np.zeros(d)
        I = self.ridge * np.eye(d)
        self.n_ll_ = []
        for _ in range(self.n_iter):
            p = self._sigmoid(Xb @ self.w)
            g = Xb.T @ (p - y) + self.ridge * self.w          # gradient
            W = p * (1.0 - p)                                  # IRLS weights
            H = Xb.T @ (Xb * W[:, None]) + I                   # Hessian
            self.w = self.w - np.linalg.solve(H, g)           # Newton step
            # track loss to show monotone, fast convergence
            p = self._sigmoid(Xb @ self.w)
            eps = 1e-12
            nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.n_ll_.append(nll)
        return self

    def predict_proba(self, X):
        return self._sigmoid(self._add_bias(np.asarray(X, dtype=float)) @ self.w)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def make_blobs(n, seed_offset=0):
    """Two Gaussian classes planted around opposite corners (recoverable signal)."""
    half = n // 2
    c0 = np.random.randn(half, 3) * 0.9 + np.array([-2.0, -1.5, 1.0])
    c1 = np.random.randn(n - half, 3) * 0.9 + np.array([2.0, 1.5, -1.0])
    X = np.vstack([c0, c1])
    y = np.hstack([np.zeros(half), np.ones(n - half)])
    idx = np.random.permutation(n)
    return X[idx], y[idx]


if __name__ == "__main__":
    np.random.seed(0)

    # ---- Exact check: Newton finds the minimizer of a quadratic in ONE step ----
    A = np.array([[3.0, 0.5, 0.0],
                  [0.5, 2.0, 0.3],
                  [0.0, 0.3, 1.5]])
    b = np.array([1.0, -2.0, 0.5])
    # f(x) = 0.5 x^T A x - b^T x  ->  grad = A x - b, hess = A, min at x* = A^-1 b
    x_star = np.linalg.solve(A, b)
    x_hat, steps = newton_minimize(lambda x: A @ x - b, lambda x: A, x0=np.zeros(3))
    print("Quadratic min -- Newton steps:", steps, "(exact solution needs 1)")
    print("  max |x_hat - x*|:", round(float(np.max(np.abs(x_hat - x_star))), 12))

    # ---- Classification: Newton-trained logistic regression vs majority baseline ----
    X_tr, y_tr = make_blobs(400)
    X_te, y_te = make_blobs(200)

    clf = NewtonLogisticRegression(n_iter=15).fit(X_tr, y_tr)
    acc = float(np.mean(clf.predict(X_te) == y_te))

    majority = float(max(y_tr.mean(), 1 - y_tr.mean()))  # predict most-common class
    base_acc = float(np.mean(np.full_like(y_te, round(y_tr.mean())) == y_te))

    print("\nNegative log-likelihood by Newton iteration (first 6):")
    print("  ", np.round(clf.n_ll_[:6], 5))
    print("Converged NLL:", round(clf.n_ll_[-1], 6))
    print("\nTest accuracy (Newton logistic): {:.3f}".format(acc))
    print("Baseline accuracy (majority class): {:.3f}".format(base_acc))
    print("Improvement over baseline: +{:.3f}".format(acc - base_acc))
    print("RESULT:", "PASS -- beats baseline" if acc > base_acc + 0.15 else "FAIL")
