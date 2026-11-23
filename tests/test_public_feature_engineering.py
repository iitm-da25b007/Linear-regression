import os
import unittest
import pandas as pd
import numpy as np

from src.feature_engineering import TaxiFeaturePipeline


class TestDynamicFeatureEngineeringCorrelation(unittest.TestCase):

    def setUp(self):
        """Locates dataset and sets up raw baseline inputs."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_path = os.path.join(project_root, "data", "taxifares.csv")

        self.assertTrue(
            os.path.exists(self.data_path),
            msg=f"Dataset file not found at expected path: {self.data_path}"
        )

        self.df = pd.read_csv(self.data_path)
        self.target_col = "fare_amount"
        self.y = self.df[self.target_col].values.astype(np.float64)

        # Baseline raw numerical features before any student engineering
        self.raw_feature_names = [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]

    @staticmethod
    def _compute_abs_correlations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates absolute Pearson correlation coefficient for each feature column against target y."""
        y_flat = y.ravel()
        correlations = []
        for col_idx in range(X.shape[1]):
            x_col = X[:, col_idx]
            # Handle zero-variance / constant features safely
            if np.std(x_col) == 0:
                correlations.append(0.0)
            else:
                corr = np.corrcoef(x_col, y_flat)[0, 1]
                correlations.append(abs(float(corr)))
        return np.array(correlations)

    def test_top3_engineered_correlations_dynamic(self):
        """Executes student pipeline dynamically, discovers feature names, and checks top 3 correlations."""
        # 1. Compute baseline correlations on raw features
        X_raw = self.df[self.raw_feature_names].values.astype(np.float64)
        raw_corrs = self._compute_abs_correlations(X_raw, self.y)
        raw_corr_pairs = sorted(
            zip(self.raw_feature_names, raw_corrs),
            key=lambda item: item[1],
            reverse=True
        )

        # 2. Run student's feature pipeline on raw data
        pipeline = TaxiFeaturePipeline()
        engineered_output = pipeline.fit_transform(self.df)

        # Dynamically extract feature matrix and column names depending on return type
        if isinstance(engineered_output, pd.DataFrame):
            X_engineered = engineered_output.values.astype(np.float64)
            eng_feature_names = list(engineered_output.columns)
        else:
            X_engineered = np.asarray(engineered_output, dtype=np.float64)
            # Inspect optional name attributes provided by student's class
            if hasattr(pipeline, "get_feature_names_out"):
                eng_feature_names = list(pipeline.get_feature_names_out())
            elif hasattr(pipeline, "feature_names"):
                eng_feature_names = list(pipeline.feature_names)
            else:
                eng_feature_names = [f"feature_{i}" for i in range(X_engineered.shape[1])]

        # Basic validity assertions on student's output matrix
        self.assertEqual(
            X_engineered.shape[0], len(self.df),
            msg="Engineered feature matrix row count does not match input dataset size."
        )
        self.assertFalse(
            np.isnan(X_engineered).any(),
            msg="Engineered feature matrix contains NaN values."
        )

        # 3. Compute correlations on student's engineered feature matrix
        eng_corrs = self._compute_abs_correlations(X_engineered, self.y)
        eng_corr_pairs = sorted(
            zip(eng_feature_names, eng_corrs),
            key=lambda item: item[1],
            reverse=True
        )

        # Display Top 3 features dynamically
        print("\n--- Baseline Top 3 Raw Feature Correlations ---")
        for name, corr in raw_corr_pairs[:3]:
            print(f"{name:<25}: {corr:.4f}")

        print("\n--- Student Top 3 Engineered Feature Correlations ---")
        for name, corr in eng_corr_pairs[:3]:
            print(f"{name:<25}: {corr:.4f}")

        top_raw_corr = raw_corr_pairs[0][1]
        top_eng_corr = eng_corr_pairs[0][1]

        # Assert 1: The best engineered feature achieves a higher or equal correlation compared to raw features
        self.assertGreaterEqual(
            top_eng_corr,
            top_raw_corr,
            msg=f"Top engineered feature correlation ({top_eng_corr:.4f}) is lower than baseline raw feature ({top_raw_corr:.4f})."
        )

        # Assert 2: Check that at least 3 feature columns were generated and evaluated
        self.assertGreaterEqual(
            len(eng_corr_pairs), 3,
            msg="Feature engineering pipeline produced fewer than 3 features."
        )


if __name__ == "__main__":
    unittest.main()
