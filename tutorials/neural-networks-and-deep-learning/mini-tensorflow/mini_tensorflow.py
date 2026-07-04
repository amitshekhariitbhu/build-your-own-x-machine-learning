import numpy as np

# A tiny TensorFlow-like framework.  Every value is a Tensor node in a
# computation graph.  Ops record how they were produced (parents + a local
# backward rule), so calling .backward() on a scalar loss runs reverse-mode
# autodiff: a topological sort of the graph, then the chain rule in reverse.


class Tensor:
    """A node in the computation graph holding a value and its gradient."""

    def __init__(self, data, parents=(), backward=None, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = parents            # upstream Tensors this depends on
        self._backward = backward or (lambda: None)  # push grad to parents
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    # --- graph builders (ops).  Each returns a new Tensor + local backward. ---
    def __add__(self, other):
        other = _wrap(other)
        out = Tensor(self.data + other.data, (self, other))

        def back():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = back
        return out

    def __mul__(self, other):
        other = _wrap(other)
        out = Tensor(self.data * other.data, (self, other))

        def back():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = back
        return out

    def matmul(self, other):
        other = _wrap(other)
        out = Tensor(self.data @ other.data, (self, other))

        def back():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = back
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))

        def back():
            self.grad += out.grad * (self.data > 0)

        out._backward = back
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,))

        def back():
            self.grad += out.grad * s * (1 - s)

        out._backward = back
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,))

        def back():
            self.grad += out.grad * np.ones_like(self.data)

        out._backward = back
        return out

    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    __matmul__ = matmul
    __radd__ = __add__
    __rmul__ = __mul__

    def __sub__(self, other):
        return self + _wrap(other) * -1.0

    def backward(self):
        # Topological order so every node is visited after its consumers.
        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                build(p)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)     # d(loss)/d(loss) = 1
        for t in reversed(topo):
            t._backward()


def _wrap(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def _unbroadcast(grad, shape):
    # Reverse numpy broadcasting so a parent gets a grad of its own shape.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# --- Loss ops built from primitives (kept as functions for clarity). ---
def mse(pred, target):
    diff = pred - _wrap(target)
    return (diff * diff).mean()


def binary_cross_entropy(prob, target):
    # -[y*log p + (1-y)*log(1-p)] averaged over the batch.
    y = _wrap(target)
    logp = _log(prob)
    log1mp = _log(_wrap(1.0) - prob)
    return ((y * logp + (_wrap(1.0) - y) * log1mp).mean()) * -1.0


def _log(t):
    x = np.clip(t.data, 1e-12, None)          # guard log/grad from zeros
    out = Tensor(np.log(x), (t,))

    def back():
        t.grad += out.grad / x

    out._backward = back
    return out


class Parameter(Tensor):
    """A trainable Tensor (leaf that accumulates gradients)."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class SGD:
    """Vanilla gradient descent over a list of Parameters."""

    def __init__(self, params, lr=0.1):
        self.params, self.lr = params, lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


if __name__ == "__main__":
    np.random.seed(0)

    # Tiny synthetic binary-classification data (2 gaussian blobs).
    n, dim, hidden = 120, 2, 8
    Xa = np.random.randn(n // 2, dim) + np.array([2.0, 2.0])
    Xb = np.random.randn(n // 2, dim) + np.array([-2.0, -2.0])
    X = Tensor(np.vstack([Xa, Xb]))
    y = np.array([[1.0]] * (n // 2) + [[0.0]] * (n // 2))

    # 2-layer net built entirely from framework ops: X -> ReLU -> sigmoid.
    W1 = Parameter(np.random.randn(dim, hidden) * np.sqrt(2.0 / dim))
    b1 = Parameter(np.zeros(hidden))
    W2 = Parameter(np.random.randn(hidden, 1) * np.sqrt(2.0 / hidden))
    b2 = Parameter(np.zeros(1))
    params = [W1, b1, W2, b2]
    opt = SGD(params, lr=0.2)

    def model(x):
        h = (x @ W1 + b1).relu()
        return (h @ W2 + b2).sigmoid()

    losses = []
    for epoch in range(150):
        opt.zero_grad()
        prob = model(X)                       # forward: builds the graph
        loss = binary_cross_entropy(prob, y)
        loss.backward()                       # reverse-mode autodiff
        opt.step()
        losses.append(float(loss.data))

    pred = (model(X).data > 0.5).astype(float)
    acc = float(np.mean(pred == y))

    print("Start loss: {:.4f}".format(losses[0]))
    print("Final loss: {:.4f}".format(losses[-1]))
    print("Loss reduction: {:.1f}x".format(losses[0] / losses[-1]))
    print("Train accuracy: {:.3f}".format(acc))
