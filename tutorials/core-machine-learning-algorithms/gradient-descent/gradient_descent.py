import numpy as np


def gradient_descent(grad_fn, init, lr=0.1, n_iter=1000, loss_fn=None):
    """Generic batch gradient descent.

    grad_fn(x) -> gradient of the objective at x.
    init       -> starting point (array-like).
    Returns the minimizer and a list of loss (or gradient-norm) values.
    """
    x = np.array(init, dtype=float)
    history = []
    for _ in range(n_iter):
        g = np.asarray(grad_fn(x), dtype=float)
        # Step downhill along the negative gradient.
        x = x - lr * g
        # Track progress: true loss if given, else gradient magnitude.
        history.append(loss_fn(x) if loss_fn else np.linalg.norm(g))
    return x, history


if __name__ == "__main__":
    np.random.seed(0)

    # --- Demo 1: minimize f(x,y) = (x-3)^2 + (y+1)^2, optimum at (3, -1) ---
    grad = lambda p: np.array([2 * (p[0] - 3), 2 * (p[1] + 1)])
    loss = lambda p: (p[0] - 3) ** 2 + (p[1] + 1) ** 2

    xy, hist = gradient_descent(grad, init=[0.0, 0.0], lr=0.1, n_iter=200, loss_fn=loss)
    print("Demo 1: quadratic bowl")
    print("  converged to:", np.round(xy, 4), "(true optimum: [3, -1])")
    print("  final loss:  {:.3e}".format(hist[-1]))

    # --- Demo 2: 1D linear regression y = w*x + b fit by gradient descent ---
    X = 2 * np.random.rand(100)
    true_w, true_b = 3.0, 4.0
    y = true_w * X + true_b + 0.1 * np.random.randn(100)
    n = len(X)

    def lin_grad(p):
        w, b = p
        err = (w * X + b) - y
        return np.array([2 * np.mean(err * X), 2 * np.mean(err)])

    lin_loss = lambda p: np.mean(((p[0] * X + p[1]) - y) ** 2)

    params, lhist = gradient_descent(lin_grad, init=[0.0, 0.0], lr=0.1, n_iter=2000, loss_fn=lin_loss)
    print("Demo 2: linear regression")
    print("  learned w={:.4f}, b={:.4f} (true w={}, b={})".format(params[0], params[1], true_w, true_b))
    print("  final MSE: {:.6f}".format(lhist[-1]))
