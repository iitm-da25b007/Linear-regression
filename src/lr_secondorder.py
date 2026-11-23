import numpy as np
from typing import List, Optional, Tuple


class NewtonRegression:
    """
    Multivariate Linear Regression fitted using a Newton-style Hessian solve
    Solves the linear system (X^T X) w = X^T y iteratively.
    """

    def __init__(
        self,
        # Standard Optimization Arguments (Kept for interface consistency)
        learning_rate: float = 1.0,  # Kept for signature compatibility
        epochs: int = 50,  # Maximum iterations
        batch_size: int = 1,  # Kept for signature compatibility
        random_state: Optional[int] = None,
        # Second Order Model Specific Arguments
        #
    ):
        # General training hyperparameters
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        # Second Order Model hyperparameter
        #

        # Model parameters
        self.weights = None
        self.bias = 0.0

        # Metric tracking
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates Mean Squared Error (MSE) across the dataset."""
        return float(np.mean((y_true - y_pred) ** 2))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Fits multivariate linear regression model using Newton-style Hessian updates,
        monitoring loss history per iteration (epoch).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize standard model parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

        for _ in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            y_pred = self.predict(X_shuffled)
            error = y_shuffled - y_pred

            dw = (-2.0 / n_samples) * np.dot(X_shuffled.T, error)

            Hw = (2.0 / n_samples) * np.dot(X_shuffled.T, X_shuffled)

            change_w = np.linalg.solve(Hw, -dw)
            change_b = np.mean(error)

            self.weights += self.lr * change_w
            self.bias += self.lr * change_b
            # Record overall dataset MSE loss at epoch/iteration completion
            full_y_pred = self.predict(X)
            epoch_loss = self.compute_loss(y, full_y_pred)
            self.train_loss_history.append(epoch_loss)

            # Record validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_loss = self.compute_loss(y_val, y_val_pred)
                self.val_loss_history.append(val_loss)

        return self.train_loss_history, self.val_loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts target variable for input feature matrix X."""
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        return np.dot(X, self.weights) + self.bias


# Backward-compatible name used by the original assignment tests.
SecondOrderMethodRegression = NewtonRegression

