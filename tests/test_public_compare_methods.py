import unittest
import numpy as np

from src.lr_sgd import SGDRegression
from src.lr_firstorder import AdamRegression
from src.lr_secondorder import NewtonRegression


class TestRegressionOptimizationModels(unittest.TestCase):

    def setUp(self):
        """Generates a reproducible synthetic regression dataset (y = Xw + b + noise)."""
        np.random.seed(42)
        n_samples, n_features = 200, 4
        X = np.random.randn(n_samples, n_features)
        true_weights = np.array([[2.5], [-1.2], [0.8], [3.0]])
        true_bias = 1.5
        noise = np.random.randn(n_samples, 1) * 0.1
        y = np.dot(X, true_weights) + true_bias + noise

        # Train / validation split
        split_idx = int(0.8 * n_samples)
        self.X_train, self.X_val = X[:split_idx], X[split_idx:]
        self.y_train, self.y_val = y[:split_idx], y[split_idx:]

    def test_prediction_output_shape_and_validity(self):
        """Verify output shapes and ensure model predictions contain no NaN/Inf values."""
        models = {
            "SGD": SGDRegression(learning_rate=0.01, epochs=10, random_state=42),
            "Adam": AdamRegression(learning_rate=0.1, epochs=10, random_state=42),
            "Newton": NewtonRegression(epochs=10, random_state=42),
        }

        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_val)

            self.assertEqual(
                preds.shape, 
                (len(self.X_val), 1), 
                msg=f"{name} output shape mismatch."
            )
            self.assertFalse(
                np.isnan(preds).any(), 
                msg=f"{name} produced NaN predictions."
            )
            self.assertFalse(
                np.isinf(preds).any(), 
                msg=f"{name} produced Inf predictions."
            )

    def test_monotone_loss_reduction(self):
        """Ensure all three optimization models reduce training MSE loss over epochs."""
        models = [
            SGDRegression(learning_rate=0.01, epochs=40, batch_size=16, random_state=42),
            AdamRegression(learning_rate=0.2, epochs=40, batch_size=16, random_state=42),
            NewtonRegression(epochs=40, random_state=42),
        ]

        for model in models:
            train_loss, _ = model.fit(self.X_train, self.y_train)
            model_name = model.__class__.__name__

            self.assertLess(
                train_loss[-1], 
                train_loss[0], 
                msg=f"{model_name} failed to minimize MSE training loss."
            )

    def test_comparative_convergence_speed(self):
        """
        On a noise-free quadratic system, Newton-style Hessian solve should achieve near-zero 
        loss faster than SGD and Adam due to exact linear system optimization.
        """
        np.random.seed(42)
        X = np.random.randn(100, 3)
        w_true = np.array([[1.5], [-2.0], [0.5]])
        y = np.dot(X, w_true) + 0.5  # Pure deterministic linear relationship

        max_epochs = 10

        sgd = SGDRegression(learning_rate=0.01, epochs=max_epochs, batch_size=10, random_state=42)
        adam = AdamRegression(learning_rate=0.3, epochs=max_epochs, batch_size=10, random_state=42)
        newton = NewtonRegression(epochs=max_epochs, random_state=42)

        sgd_loss, _ = sgd.fit(X, y)
        adam_loss, _ = adam.fit(X, y)
        newton_loss, _ = newton.fit(X, y)

        # Newton-style Hessian solve should reach exact solution within N=4 dimensions (3 features + 1 bias)
        self.assertLess(
            newton_loss[-1], 
            1e-5, 
            msg=f"Newton-style update failed exact convergence. Final loss: {newton_loss[-1]}"
        )
        
        # Assert performance superiority of Newton-style update over standard first-order updates in equal low epochs
        self.assertLess(
            newton_loss[-1], 
            sgd_loss[-1], 
            msg="Newton-style Hessian solve should outperform SGD in few-epoch convergence."
        )
        self.assertLess(
            newton_loss[-1], 
            adam_loss[-1], 
            msg="Newton-style Hessian solve should outperform Adam in few-epoch convergence."
        )

    def test_validation_loss_tracking(self):
        """Confirm validation loss history is logged correctly across all iterations when provided."""
        models = [
            SGDRegression(epochs=5, random_state=42),
            AdamRegression(epochs=5, random_state=42),
            NewtonRegression(epochs=5, random_state=42),
        ]

        for model in models:
            train_hist, val_hist = model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
            model_name = model.__class__.__name__

            self.assertEqual(
                len(train_hist), 
                5, 
                msg=f"{model_name} train loss length mismatch."
            )
            self.assertEqual(
                len(val_hist), 
                5, 
                msg=f"{model_name} val loss length mismatch."
            )


if __name__ == "__main__":
    unittest.main()


