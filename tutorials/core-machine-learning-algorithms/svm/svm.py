import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # L2 regularization strength
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Map labels {0, 1} (or any 2 classes) to {-1, +1}
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Sub-gradient descent on hinge loss + L2 regularization
        for _ in range(self.n_epochs):
            for i in range(n_samples):
                margin = y_[i] * (np.dot(X[i], self.weights) - self.bias)
                if margin >= 1:
                    # Correctly classified beyond margin: only regularization gradient
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # Inside margin / misclassified: add hinge loss gradient
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - y_[i] * X[i])
                    self.bias -= self.learning_rate * y_[i]

    def predict(self, X):
        # Decision rule: sign(w·x - b), returned as {-1, +1}
        linear_model = np.dot(X, self.weights) - self.bias
        return np.sign(linear_model)

# Example usage
if __name__ == "__main__":
    # Generate linearly separable 2-class data
    np.random.seed(0)
    X_pos = np.random.randn(50, 2) + np.array([2, 2])   # class +1 cluster
    X_neg = np.random.randn(50, 2) + np.array([-2, -2])  # class -1 cluster
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(50), -np.ones(50)))

    # Train the model
    model = SVM(learning_rate=0.001, lambda_param=0.01, n_epochs=1000)
    model.fit(X, y)

    # Evaluate on the training data
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)

    # Print results
    print("Learned weights:", model.weights)
    print("Learned bias:", model.bias)
    print("Accuracy:", accuracy)
