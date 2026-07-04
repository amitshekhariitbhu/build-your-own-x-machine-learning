import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Calculate mean, variance, and prior probability for each class
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9  # epsilon avoids division by zero
            self.priors[idx] = X_c.shape[0] / n_samples

    def _gaussian_pdf(self, class_idx, x):
        # Gaussian probability density function for each feature
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_single(self, x):
        posteriors = []

        # Combine log prior and log likelihood for each class
        for idx in range(len(self.classes)):
            log_prior = np.log(self.priors[idx])
            log_likelihood = np.sum(np.log(self._gaussian_pdf(idx, x)))
            posteriors.append(log_prior + log_likelihood)

        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])


# Example usage
if __name__ == "__main__":
    # Generate synthetic data: two Gaussian clusters (binary classification)
    np.random.seed(0)
    X_class0 = np.random.randn(50, 2) + np.array([2, 2])    # Class 0 around (2, 2)
    X_class1 = np.random.randn(50, 2) + np.array([-2, -2])  # Class 1 around (-2, -2)
    X = np.vstack((X_class0, X_class1))
    y = np.array([0] * 50 + [1] * 50)

    # Create and train model
    model = GaussianNaiveBayes()
    model.fit(X, y)

    # Make predictions
    test_points = np.array([[1.5, 2.0], [-2.0, -1.5], [0.0, 0.0]])
    predictions = model.predict(test_points)

    # Evaluate on training data
    accuracy = np.mean(model.predict(X) == y)

    # Print results
    print("Test points:", test_points.tolist())
    print("Predictions:", predictions)
    print("Class priors:", model.priors)
    print("Class means:\n", model.mean)
    print("Training accuracy:", accuracy)
