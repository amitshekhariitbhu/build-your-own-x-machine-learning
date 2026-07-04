import numpy as np


class SGD:
    """Stochastic gradient descent with optional momentum."""

    def __init__(self, lr=0.1, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        # Velocity blends past step with current gradient.
        self.v = self.momentum * self.v - self.lr * grads
        return params + self.v


class Adagrad:
    """Adagrad: scale learning rate by accumulated squared gradients."""

    def __init__(self, lr=0.5, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.G = None

    def step(self, params, grads):
        if self.G is None:
            self.G = np.zeros_like(params)
        # Accumulate squared gradients (grows monotonically).
        self.G += grads ** 2
        return params - self.lr * grads / (np.sqrt(self.G) + self.eps)


class RMSprop:
    """RMSprop: exponential moving average of squared gradients."""

    def __init__(self, lr=0.1, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = None

    def step(self, params, grads):
        if self.s is None:
            self.s = np.zeros_like(params)
        # Decaying average keeps the scale from exploding.
        self.s = self.beta * self.s + (1 - self.beta) * grads ** 2
        return params - self.lr * grads / (np.sqrt(self.s) + self.eps)


class Adam:
    """Adam: bias-corrected first and second moment estimates."""

    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        # First moment (mean) and second moment (variance) of gradients.
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        # Correct the zero-initialization bias early in training.
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


if __name__ == "__main__":
    np.random.seed(0)

    # Convex objective f(w) = sum((w - target)^2), minimized at w = target.
    target = np.array([3.0, -2.0, 5.0])
    grad = lambda w: 2 * (w - target)

    optimizers = {
        "SGD(momentum=0.9)": SGD(lr=0.1, momentum=0.9),
        "Adagrad": Adagrad(lr=1.0),
        "RMSprop": RMSprop(lr=0.1),
        "Adam": Adam(lr=0.2),
    }

    print("Target w:", target, "\n")
    for name, opt in optimizers.items():
        w = np.zeros(3)  # start far from the optimum
        for _ in range(500):
            w = opt.step(w, grad(w))
        err = np.linalg.norm(w - target)
        print("{:20s} final w = {}  |w-target| = {:.2e}".format(
            name, np.round(w, 4), err))
