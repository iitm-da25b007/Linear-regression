import os
import json
import unittest
import pandas as pd
import numpy as np

from src.feature_engineering import TaxiFeaturePipeline
from src.lr_sgd import SGDRegression
from src.lr_firstorder import AdamRegression
from src.lr_secondorder import NewtonRegression


class TestTaxiFaresPipeline(unittest.TestCase):

    def setUp(self):
        """Locates dataset, creates results folder, runs feature engineering pipeline, and prepares train/val splits."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_path = os.path.join(project_root, "data", "taxifares.csv")
        
        # Ensure results directory exists
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        self.output_json_path = os.path.join(results_dir, "validation_loss_comparison.json")

        self.assertTrue(
            os.path.exists(self.data_path),
            msg=f"Dataset file not found at expected path: {self.data_path}"
        )

        df = pd.read_csv(self.data_path)

        # 80/20 Train / Validation split on DataFrame level
        n_samples = len(df)
        split_idx = int(0.8 * n_samples)
        df_train = df.iloc[:split_idx].copy()
        df_val = df.iloc[split_idx:].copy()

        # Extract targets
        self.y_train = df_train[["fare_amount"]].values.astype(np.float64)
        self.y_val = df_val[["fare_amount"]].values.astype(np.float64)

        # Feature engineering pipeline
        self.pipeline = TaxiFeaturePipeline()
        train_engineered = self.pipeline.fit_transform(df_train)
        val_engineered = self.pipeline.transform(df_val)

        self.X_train = (
            train_engineered.values.astype(np.float64)
            if isinstance(train_engineered, pd.DataFrame)
            else np.asarray(train_engineered, dtype=np.float64)
        )
        self.X_val = (
            val_engineered.values.astype(np.float64)
            if isinstance(val_engineered, pd.DataFrame)
            else np.asarray(val_engineered, dtype=np.float64)
        )

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculates MSE, RMSE, MAE, R2 Score, and MAPE for model predictions."""
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()

        mse = float(np.mean((y_true_flat - y_pred_flat) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))

        # Coefficient of Determination (R^2 Score)
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

        # Mean Absolute Percentage Error (MAPE)
        nonzero_mask = y_true_flat != 0
        if np.any(nonzero_mask):
            mape = float(np.mean(np.abs((y_true_flat[nonzero_mask] - y_pred_flat[nonzero_mask]) / y_true_flat[nonzero_mask])) * 100.0)
        else:
            mape = 0.0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
            "mape_percent": mape
        }

    def test_compare_first_vs_second_order_methods(self):
        """
        Trains models for extended epochs (100) and compares convergence, MSE, R2, and MAPE
        between First-Order (SGD, Adam) and Second-Order (Newton-style Hessian solve) methods.
        """
        epochs = 100
        batch_size = 64

        first_order_models = {
            "SGD": SGDRegression(learning_rate=0.01, epochs=epochs, batch_size=batch_size, random_state=42),
            "Adam": AdamRegression(learning_rate=0.1, epochs=epochs, batch_size=batch_size, random_state=42),
        }

        second_order_models = {
            "Newton": NewtonRegression(epochs=epochs, random_state=42)
        }

        first_order_results = {}
        second_order_results = {}

        # 1. Evaluate First-Order Methods
        for name, model in first_order_models.items():
            model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
            y_val_pred = model.predict(self.X_val)
            
            metrics = self._compute_metrics(self.y_val, y_val_pred)
            self.assertFalse(np.isnan(metrics["mse"]), msg=f"{name} produced NaN validation loss.")
            self.assertLess(metrics["mse"], 1e5, msg=f"{name} validation loss diverged.")
            
            metrics["epochs_trained"] = epochs
            first_order_results[name] = metrics

        # 2. Evaluate Second-Order Methods
        for name, model in second_order_models.items():
            model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
            y_val_pred = model.predict(self.X_val)
            
            metrics = self._compute_metrics(self.y_val, y_val_pred)
            self.assertFalse(np.isnan(metrics["mse"]), msg=f"{name} produced NaN validation loss.")
            self.assertLess(metrics["mse"], 1e5, msg=f"{name} validation loss diverged.")
            
            metrics["epochs_trained"] = epochs
            second_order_results[name] = metrics

        # Baseline metrics (mean predictor baseline)
        y_baseline_pred = np.full_like(self.y_val, np.mean(self.y_train))
        baseline_metrics = self._compute_metrics(self.y_val, y_baseline_pred)

        # Comparative Analytics
        best_first_order_mse = min(res["mse"] for res in first_order_results.values())
        best_second_order_mse = min(res["mse"] for res in second_order_results.values())

        # Construct comparative JSON payload
        results_payload = {
            "training_config": {
                "epochs": epochs,
                "batch_size": batch_size
            },
            "baseline_val_metrics": baseline_metrics,
            "first_order_methods": first_order_results,
            "second_order_methods": second_order_results,
            "comparison": {
                "best_first_order_mse": best_first_order_mse,
                "best_second_order_mse": best_second_order_mse,
                "second_order_advantage_mse": best_first_order_mse - best_second_order_mse
            }
        }

        # Write comparative metrics to JSON
        with open(self.output_json_path, "w") as f:
            json.dump(results_payload, f, indent=4)

        self.assertTrue(
            os.path.exists(self.output_json_path),
            msg=f"Failed to create output JSON file at {self.output_json_path}"
        )

        print(f"\nExtended training complete ({epochs} epochs).")
        print(f"Best First-Order Val MSE  : {best_first_order_mse:.4f}")
        print(f"Best Second-Order Val MSE : {best_second_order_mse:.4f}")
        print(f"Saved comparative metrics (including R2 & MAPE) to: {self.output_json_path}")


if __name__ == "__main__":
    unittest.main()

