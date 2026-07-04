import numpy as np


class Tensor:
    """A PyTorch-like tensor with tape-based reverse-mode autodiff (define-by-run).

    Each op builds a node that remembers its parents and a local _backward
    closure. Calling .backward() walks the tape in reverse-topological order,
    seeding the output gradient with 1 and applying the chain rule.
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)   # accumulated dLoss/dself
        self._backward = lambda: None          # local vjp for this node
        self._prev = set(_children)            # parents on the tape
        self._op = _op

    @staticmethod
    def _reduce(grad, shape):
        # Sum a gradient back down to `shape`, undoing numpy broadcasting.
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = self.grad + self._reduce(out.grad, self.data.shape)
            other.grad = other.grad + self._reduce(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = self.grad + self._reduce(out.grad * other.data, self.data.shape)
            other.grad = other.grad + self._reduce(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad = self.grad + out.grad @ other.data.T
            other.grad = other.grad + self.data.T @ out.grad
        out._backward = _backward
        return out

    def __pow__(self, p):
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad = self.grad + (p * self.data ** (p - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad = self.grad + (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad = self.grad + np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(self.data.mean(), (self,), "mean")

        def _backward():
            self.grad = self.grad + np.ones_like(self.data) * out.grad / self.data.size
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-(other if isinstance(other, Tensor) else Tensor(other)))

    __radd__ = __add__
    __rmul__ = __mul__

    def backward(self):
        # Reverse-topological order over the tape, then apply each local vjp.
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = np.ones_like(self.data)     # seed dLoss/dLoss = 1
        for v in reversed(topo):
            v._backward()


# --- nn-style building blocks ---

class Linear:
    """Fully-connected layer: y = x @ W + b."""

    def __init__(self, n_in, n_out):
        self.W = Tensor(np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in))
        self.b = Tensor(np.zeros(n_out))

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP:
    """Stack of Linear layers with ReLU between hidden layers."""

    def __init__(self, sizes):
        self.layers = [Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class SGD:
    """Vanilla stochastic gradient descent optimizer."""

    def __init__(self, params, lr=0.1):
        self.params, self.lr = params, lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


def mse_loss(pred, target):
    # Mean squared error, built purely from tensor ops so it flows on the tape.
    return ((pred - target) ** 2).mean()


if __name__ == "__main__":
    np.random.seed(0)

    # --- Autograd sanity: f = a*b + a**2 ; df/da = b + 2a, df/db = a ---
    a, b = Tensor(3.0), Tensor(4.0)
    f = a * b + a ** 2
    f.backward()
    print("df/da = {:.1f} (expect {:.1f})".format(float(a.grad), 4 + 2 * 3))
    print("df/db = {:.1f} (expect {:.1f})".format(float(b.grad), 3.0))

    # --- Tiny synthetic 3-class blobs dataset ---
    n_per, n_class, dim = 40, 3, 4
    centers = np.random.randn(n_class, dim) * 3.0
    X = np.vstack([centers[c] + np.random.randn(n_per, dim) for c in range(n_class)])
    y = np.repeat(np.arange(n_class), n_per)
    X = (X - X.mean(axis=0)) / X.std(axis=0)     # standardize features
    Y = np.eye(n_class)[y]                        # one-hot targets

    Xt, Yt = Tensor(X), Tensor(Y)
    model = MLP([dim, 16, n_class])
    opt = SGD(model.parameters(), lr=0.2)

    # --- Train the MLP entirely through tensor.backward() ---
    losses = []
    for epoch in range(300):
        opt.zero_grad()
        loss = mse_loss(model(Xt), Yt)           # define-by-run forward
        loss.backward()                          # reverse-mode autodiff
        opt.step()
        losses.append(float(loss.data))

    acc = np.mean(model(Xt).data.argmax(axis=1) == y)
    print("Start loss:     {:.4f}".format(losses[0]))
    print("Final loss:     {:.4f}".format(losses[-1]))
    print("Loss reduction: {:.1f}x".format(losses[0] / losses[-1]))
    print("Train accuracy: {:.3f}".format(acc))
