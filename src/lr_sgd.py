import numpy as np
from typing import List, Optional, Tuple

class SGDRegression:
    """
    Multivariate Linear Regression fitted using Stochastic Gradient Descent (SGD).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 1,
        random_state: Optional[int] = None,
    ):
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size  # batch_size=1 represents pure SGD
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0
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
        Fits multivariate linear regression model using Stochastic Gradient Descent,
        updating parameters per sample/mini-batch and monitoring loss history per epoch.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []

        for _ in range(self.epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Process individual samples or mini-batches
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]

                # Prediction and error for current sample/batch
                y_pred_batch = np.dot(X_batch, self.weights) + self.bias
                error = y_batch - y_pred_batch  # Shape: (batch_size, 1)

                # Stochastic gradient computation
                dw = (-2.0 / len(X_batch)) * np.dot(X_batch.T, error)
                db = (-2.0 / len(X_batch)) * np.sum(error)

                # Parameter update step immediately after each sample/batch
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            # Log overall dataset MSE loss at epoch completion
            full_y_pred = np.dot(X, self.weights) + self.bias
            epoch_loss = self.compute_loss(y, full_y_pred)
            self.train_loss_history.append(epoch_loss)

            # Record validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                y_val_pred = np.dot(X_val, self.weights) + self.bias
                val_loss = self.compute_loss(y_val, y_val_pred)
                self.val_loss_history.append(val_loss)

        return self.train_loss_history, self.val_loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts target variable for input feature matrix X."""
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        return np.dot(X, self.weights) + self.bias
