import numpy as np

# Configurable activation functions: (forward, derivative-w.r.t.-pre-activation)
def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(z.dtype)

def tanh(z):
    return np.tanh(z)

def tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "tanh": (tanh, tanh_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
}

def softmax(z):
    # Numerically stable softmax over the last axis
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

class FeedforwardNeuralNetwork:
    """General feedforward net: any number of layers + configurable activation.
    Hidden layers use `activation`; the output layer uses softmax for
    multiclass classification, trained by manual backprop on cross-entropy."""

    def __init__(self, layer_sizes, activation="relu", learning_rate=0.1, n_epochs=400):
        # layer_sizes e.g. [2, 16, 8, 3]: input -> hidden... -> output(softmax)
        self.layer_sizes = layer_sizes
        self.act, self.act_grad = ACTIVATIONS[activation]
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights, self.biases = [], []
        # He-style random init for each layer
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in))
            self.biases.append(np.zeros((1, fan_out)))

    def forward(self, X):
        # Forward pass, caching pre-activations (z) and activations (a) for backprop
        self.zs, self.activations = [], [X]
        a = X
        n_layers = len(self.weights)
        for i in range(n_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.zs.append(z)
            # hidden activation on all but last layer, softmax on the output layer
            a = softmax(z) if i == n_layers - 1 else self.act(z)
            self.activations.append(a)
        return a

    def backward(self, y_onehot):
        # Manual backprop; d(softmax + cross-entropy)/dz at output is (pred - y)
        n = y_onehot.shape[0]
        n_layers = len(self.weights)
        grads_w, grads_b = [None] * n_layers, [None] * n_layers
        delta = (self.activations[-1] - y_onehot) / n
        for i in reversed(range(n_layers)):
            grads_w[i] = self.activations[i].T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                # Propagate through the hidden activation's derivative
                delta = (delta @ self.weights[i].T) * self.act_grad(self.zs[i - 1])
        for i in range(n_layers):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def loss(self, y_pred, y_onehot):
        # Categorical cross-entropy
        return -np.mean(np.sum(y_onehot * np.log(y_pred + 1e-9), axis=1))

    def _one_hot(self, y, n_classes):
        oh = np.zeros((y.shape[0], n_classes))
        oh[np.arange(y.shape[0]), y] = 1
        return oh

    def fit(self, X, y):
        # y: integer class labels shape (n,)
        y_onehot = self._one_hot(y, self.layer_sizes[-1])
        history = []
        for _ in range(self.n_epochs):
            y_pred = self.forward(X)
            history.append(self.loss(y_pred, y_onehot))
            self.backward(y_onehot)
        return history

    def predict_probability(self, X):
        return self.forward(X)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    # Tiny synthetic 3-class blobs in 2D
    n_per, n_classes = 60, 3
    centers = np.array([[-2.5, 0.0], [2.5, 0.0], [0.0, 3.0]])
    X = np.vstack([c + 0.7 * np.random.randn(n_per, 2) for c in centers])
    y = np.repeat(np.arange(n_classes), n_per)

    model = FeedforwardNeuralNetwork(layer_sizes=[2, 16, 8, 3],
                                     activation="relu",
                                     learning_rate=0.2, n_epochs=400)
    history = model.fit(X, y)

    accuracy = np.mean(model.predict(X) == y)
    print("Layer sizes: ", model.layer_sizes)
    print("Start loss:  ", round(history[0], 4))
    print("Final loss:  ", round(history[-1], 4))
    print("Train accuracy:", round(float(accuracy), 4))
