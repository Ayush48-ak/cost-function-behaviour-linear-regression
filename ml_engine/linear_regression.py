
import numpy as np


class LinearRegressionGD:
    """
    Linear Regression implemented using NumPy and Gradient Descent.
    Tracks the Mean Squared Error (MSE) during training.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.weights = None
        self.bias = 0.0
        self.feature_mean = None
        self.feature_std = None
        self.cost_history = []

    def _scale_features(self, X):
        """Standardize features using training-set statistics."""
        return (X - self.feature_mean) / self.feature_std

    def fit(self, X, y):
        """Train the model using Gradient Descent."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("X must be 2D and match the length of y.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Training data cannot be empty.")

        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Training data must contain finite values.")

        # Learn scaling parameters from training data only.
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0)
        self.feature_std[self.feature_std == 0] = 1.0

        X_scaled = self._scale_features(X)

        n_samples, n_features = X_scaled.shape

        # Initialize model parameters.
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        # Gradient Descent
        for _ in range(self.n_iterations):
            predictions = X_scaled @ self.weights + self.bias
            errors = predictions - y

            # Calculate gradients.
            dw = (2 / n_samples) * (X_scaled.T @ errors)
            db = (2 / n_samples) * np.sum(errors)

            # Update weights and bias.
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Record cost after each update.
            updated_predictions = X_scaled @ self.weights + self.bias
            cost = np.mean((updated_predictions - y) ** 2)
            self.cost_history.append(float(cost))

        return self

    def predict(self, X):
        """Predict target values for new observations."""
        if self.weights is None:
            raise ValueError("Train the model before making predictions.")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError("Input features do not match the trained model.")

        if not np.isfinite(X).all():
            raise ValueError("Prediction data must contain finite values.")

        X_scaled = self._scale_features(X)
        return X_scaled @ self.weights + self.bias

    def calculate_mse(self, y_true, y_pred):
        """Calculate Mean Squared Error."""
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

        if y_true.shape != y_pred.shape:
            raise ValueError("Actual and predicted values must have equal lengths.")

        return float(np.mean((y_true - y_pred) ** 2))
