import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        #Store the training data.
        self.X_train = X
        self.y_train = y

    def predict(self, x):
        # Calculate Euclidean distances to all training samples
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # Return most common label (majority vote)
        return np.bincount(k_nearest_labels).argmax()


# Example usage
if __name__ == "__main__":
    # Generate sample data (simple linear boundaries for 3 classes)
    np.random.seed(0)
    X = np.random.rand(150, 2) * 2  # 150 samples, 2 features in [0, 2]
    # Assign labels based on two linear boundaries
    y = np.zeros(150, dtype=int)
    y[(X[:, 0] + X[:, 1] > 1.5)] = 1  # Class 1 if x_1 + x_2 > 1.5
    y[(X[:, 0] + X[:, 1] < 0.5)] = 2  # Class 2 if x_1 + x_2 < 0.5
    # Class 0 for 0.5 <= x_1 + x_2 <= 1.5

    # Create and train model
    model = KNN(k=3)
    model.fit(X, y)

    # Predict
    test_point_1 = np.array([1.5, 0.5])  # x_1 + x_2 = 2 > 1.5, should be Class 1
    prediction_1 = model.predict(test_point_1)
    test_point_2 = np.array([0.2, 0.5])  # 0.5 <= x_1 + x_2 <= 1.5, should be class 0
    prediction_2 = model.predict(test_point_2)

    # Print results
    print("Test point 1:", test_point_1)
    print("Predicted class for Test point 1:", prediction_1)
    print("Test point 2:", test_point_2)
    print("Predicted class for Test point 2:", prediction_2)
