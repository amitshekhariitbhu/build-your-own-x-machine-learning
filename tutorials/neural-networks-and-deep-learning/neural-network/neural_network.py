import numpy as np

def sigmoid(z):
    # Sigmoid activation, mapped to (0, 1)
    return 1 / (1 + np.exp(-z))

def relu(z):
    # ReLU activation
    return np.maximum(0, z)

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1, n_epochs=3000):
        # layer_sizes e.g. [2, 4, 1]: input -> hidden(ReLU) -> output(sigmoid)
        self.layer_sizes = layer_sizes
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
            # ReLU on hidden layers, sigmoid on the output layer
            a = sigmoid(z) if i == n_layers - 1 else relu(z)
            self.activations.append(a)
        return a

    def backward(self, y):
        # Manual backprop with binary cross-entropy loss
        n = y.shape[0]
        n_layers = len(self.weights)
        grads_w = [None] * n_layers
        grads_b = [None] * n_layers
        # d(BCE + sigmoid)/dz at output simplifies to (pred - y)
        delta = (self.activations[-1] - y) / n
        for i in reversed(range(n_layers)):
            grads_w[i] = self.activations[i].T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                # Propagate through ReLU derivative of the previous hidden layer
                delta = (delta @ self.weights[i].T) * (self.zs[i - 1] > 0)
        for i in range(n_layers):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def loss(self, y_pred, y):
        # Binary cross-entropy
        eps = 1e-9
        return -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

    def fit(self, X, y):
        history = []
        for epoch in range(self.n_epochs):
            y_pred = self.forward(X)
            history.append(self.loss(y_pred, y))
            self.backward(y)
        return history

    def predict_probability(self, X):
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int)

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    # XOR problem: not linearly separable, needs a hidden layer
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    model = NeuralNetwork(layer_sizes=[2, 8, 1], learning_rate=0.5, n_epochs=3000)
    history = model.fit(X, y)

    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)

    print("Start loss:  ", round(history[0], 4))
    print("Final loss:  ", round(history[-1], 4))
    print("Predictions: ", predictions.ravel())
    print("Targets:     ", y.ravel().astype(int))
    print("XOR accuracy:", accuracy)
