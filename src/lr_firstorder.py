import numpy as np
from typing import List, Optional, Tuple


class AdamRegression:
    """
    Multivariate Linear Regression fitted using First-Order Optimization Methods
    """

    def __init__(
        self,
        # Standard First-Order Optimization Arguments
        learning_rate: float = 1.0,
        epochs: int = 50,
        batch_size: int = 1,
        random_state: Optional[int] = None,
        # First Order Model-Specific Hyperparameters
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        # General training hyperparameters
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        # First Order Model hyperparameter
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Model parameters
        self.weights = None
        self.bias = 0.0

        # First Order Model parameters
        self.m_weights = None
        self.m_bias = 0.0

        self.v_weights = None
        self.v_bias = 0.0

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
        Fits multivariate linear regression model using First Order Model updates.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize standard model parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

        # Initialize First Order Model parameters
        self.m_weights = np.zeros((n_features, 1))
        self.v_weights = np.zeros((n_features, 1))

        self.m_bias = 0.0
        self.v_bias = 0.0

        self.train_loss_history = []
        self.val_loss_history = []

        t = 0

        for _ in range(self.epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Process mini-batches
            for i in range(0, n_samples, self.batch_size):
                # implement your first order method here
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size].reshape(-1, 1)

                # Prediction and error for current sample/batch
                y_pred_batch = self.predict(X_batch)
                error = y_batch - y_pred_batch

                dw = (-2.0 / len(X_batch)) * np.dot(X_batch.T, error)
                db = (-2.0 / len(X_batch)) * np.sum(error)

                t += 1

                self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * dw
                self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * db

                self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (
                    dw**2
                )
                self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * (db**2)

                # Bias correction
                m_w = self.m_weights / (1 - self.beta1**t)
                m_b = self.m_bias / (1 - self.beta1**t)
                v_w = self.v_weights / (1 - self.beta2**t)
                v_b = self.v_bias / (1 - self.beta2**t)

                self.weights -= self.lr * m_w / (np.sqrt(v_w) + self.epsilon)
                self.bias -= self.lr * m_b / (np.sqrt(v_b) + self.epsilon)

            # Log overall dataset MSE loss at epoch completion
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
FirstOrderMethodRegression = AdamRegression

