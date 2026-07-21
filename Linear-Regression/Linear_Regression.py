import numpy as np

class LinearRegressionScratch:

    def __init__(self, learning_rate=0.01, epochs=5000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    # Training
    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        
        for _ in range(self.epochs):

            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Errors
            error = y_pred - y

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)
            

            # Update Parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # Prediction
    def predict(self, X):
        
        return np.dot(X, self.weights) + self.bias
