import numpy as np


class TwoLayerNet:
    """2-layer MLP (ReLU hidden, softmax output) with hand-derived backprop.

    Architecture:  x -> [W1,b1] -> ReLU -> h -> [W2,b2] -> softmax -> p
    Loss:          mean cross-entropy over the batch.
    """

    def __init__(self, n_in, n_hidden, n_out, reg=1e-3, seed=0):
        rng = np.random.RandomState(seed)
        # Small random init; scaling keeps activations in a sane range.
        self.params = {
            "W1": rng.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in),
            "b1": np.zeros(n_hidden),
            "W2": rng.randn(n_hidden, n_out) * np.sqrt(2.0 / n_hidden),
            "b2": np.zeros(n_out),
        }
        self.reg = reg

    def forward(self, X):
        # Layer 1 -> ReLU -> Layer 2 -> softmax.  Cache holds intermediates.
        W1, b1, W2, b2 = (self.params[k] for k in ("W1", "b1", "W2", "b2"))
        z1 = X @ W1 + b1
        h = np.maximum(0, z1)              # ReLU
        scores = h @ W2 + b2

        # Numerically stable softmax.
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        probs = exp / exp.sum(axis=1, keepdims=True)

        cache = (X, z1, h, probs)
        return probs, cache

    def loss(self, X, y):
        # Cross-entropy data loss + L2 regularization.
        probs, cache = self.forward(X)
        n = X.shape[0]
        data_loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
        reg_loss = 0.5 * self.reg * sum(
            np.sum(self.params[k] ** 2) for k in ("W1", "W2")
        )
        return data_loss + reg_loss, cache

    def backward(self, cache, y):
        # Analytic gradients via the chain rule.
        X, z1, h, probs = cache
        n = X.shape[0]
        W1, W2 = self.params["W1"], self.params["W2"]

        # d(loss)/d(scores) for softmax + cross-entropy: (probs - onehot) / n.
        dscores = probs.copy()
        dscores[np.arange(n), y] -= 1
        dscores /= n

        # Output layer.
        dW2 = h.T @ dscores + self.reg * W2
        db2 = dscores.sum(axis=0)

        # Backprop into hidden layer, through ReLU.
        dh = dscores @ W2.T
        dz1 = dh * (z1 > 0)

        # Input layer.
        dW1 = X.T @ dz1 + self.reg * W1
        db1 = dz1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def predict(self, X):
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)

    def train(self, X, y, lr=0.5, epochs=200):
        # Full-batch gradient descent.
        history = []
        for _ in range(epochs):
            loss, cache = self.loss(X, y)
            grads = self.backward(cache, y)
            for k in self.params:
                self.params[k] -= lr * grads[k]
            history.append(loss)
        return history


def numerical_gradient(net, X, y, param, h=1e-5):
    """Finite-difference (central) gradient of the loss w.r.t. one parameter."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = param[idx]
        param[idx] = old + h
        fp, _ = net.loss(X, y)
        param[idx] = old - h
        fm, _ = net.loss(X, y)
        param[idx] = old
        grad[idx] = (fp - fm) / (2 * h)
        it.iternext()
    return grad


def gradient_check(net, X, y):
    # Compare analytic vs numerical gradients; return worst relative error.
    _, cache = net.loss(X, y)
    analytic = net.backward(cache, y)
    worst = 0.0
    for k in net.params:
        num = numerical_gradient(net, X, y, net.params[k])
        ana = analytic[k]
        rel = np.abs(ana - num) / np.maximum(1e-12, np.abs(ana) + np.abs(num))
        worst = max(worst, rel.max())
    return worst


if __name__ == "__main__":
    np.random.seed(0)

    # Tiny synthetic 3-class "blobs" dataset.
    n_per, n_class, dim = 60, 3, 4
    centers = np.random.randn(n_class, dim) * 3.0
    X = np.vstack([centers[c] + np.random.randn(n_per, dim) for c in range(n_class)])
    y = np.repeat(np.arange(n_class), n_per)

    # --- Correctness signal: analytic vs numerical gradients ---
    check_net = TwoLayerNet(dim, 5, n_class, reg=1e-2, seed=1)
    err = gradient_check(check_net, X[:15], y[:15])
    print("Gradient check max relative error: {:.2e}  (expect ~1e-7)".format(err))

    # --- Training: loss must drop substantially ---
    net = TwoLayerNet(dim, 16, n_class, reg=1e-3, seed=0)
    hist = net.train(X, y, lr=0.3, epochs=200)
    acc = np.mean(net.predict(X) == y)

    print("Start loss: {:.4f}".format(hist[0]))
    print("Final loss: {:.4f}".format(hist[-1]))
    print("Loss reduction: {:.1f}x".format(hist[0] / hist[-1]))
    print("Train accuracy: {:.3f}".format(acc))
