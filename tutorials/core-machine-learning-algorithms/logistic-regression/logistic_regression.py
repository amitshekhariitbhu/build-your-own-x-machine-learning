import numpy as np

def sigmoid(z):
    # Sigmoid function to map values to (0, 1)
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Calculate parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate loss (binary cross-entropy)
            loss = -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted ))

    def predict_probability(self, X):
        # Predict probabilities
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        # Predict class labels (0 or 1) based on threshold
        probabilities = self.predict_probability(X)
        return (probabilities >= threshold).astype(int)

# Example usage
if __name__ == "__main__":
    # Generate sample data (binary classification)
    np.random.seed(0)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Class 1 if sum of features > 0, else 0

    # Create and train model
    model = LogisticRegression(learning_rate=0.1, n_epochs=1000)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict([[-0.03, -1.16], [1.49, 0.43]])

    # Print results
    print("Predictions:", predictions)
    print("Learned weights:", model.weights)
    print("Learned bias:", model.bias)
