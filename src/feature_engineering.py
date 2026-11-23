import numpy as np
import pandas as pd


class TaxiFeaturePipeline:
    """
    Feature engineering and normalization pipeline for taxi fare datasets.
    Prevents data leakage by fitting scaling parameters on train data only.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.pickup_lat_median = None
        self.pickup_lon_median = None
        self.dropoff_lat_median = None
        self.dropoff_lon_median = None

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts spatial, temporal, and numeric features from raw DataFrame."""
        df_ = df.copy()

        def haversine(lon_1, lon_2, lat_1, lat_2):
            lat1 = np.radians(lat_1)
            lat2 = np.radians(lat_2)

            dlat = lat2 - lat1
            dlon = np.radians(lon_2 - lon_1)

            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            a = np.clip(a, 0, 1)

            distance = 2 * 6371 * np.arcsin(np.sqrt(a))

            return distance

        def manhattan(lon_1, lon_2, lat_1, lat_2):
            return haversine(lon_1, lon_2, lat_1, lat_1) + haversine(
                lon_1, lon_1, lat_1, lat_2
            )

        # 1. Extract relevant features
        lat_min = 40.4
        lat_max = 41.0
        lon_min = -74.3
        lon_max = -73.6
        pickup_lat = df_["pickup_latitude"]
        pickup_lon = df_["pickup_longitude"]
        dropoff_lat = df_["dropoff_latitude"]
        dropoff_lon = df_["dropoff_longitude"]
        pickup_valid = (
            (pickup_lat >= lat_min)
            & (pickup_lat <= lat_max)
            & (pickup_lon >= lon_min)
            & (pickup_lon <= lon_max)
        )
        dropoff_valid = (
            (dropoff_lat >= lat_min)
            & (dropoff_lat <= lat_max)
            & (dropoff_lon >= lon_min)
            & (dropoff_lon <= lon_max)
        )
        if self.dropoff_lat_median is None:
            self.pickup_lat_median = np.median(pickup_lat[pickup_valid])
            self.pickup_lon_median = np.median(pickup_lon[pickup_valid])
            self.dropoff_lat_median = np.median(dropoff_lat[dropoff_valid])
            self.dropoff_lon_median = np.median(dropoff_lon[dropoff_valid])
        pickup_lat = np.where(pickup_valid, pickup_lat, self.pickup_lat_median)
        pickup_lon = np.where(pickup_valid, pickup_lon, self.pickup_lon_median)
        dropoff_lat = np.where(dropoff_valid, dropoff_lat, self.dropoff_lat_median)
        dropoff_lon = np.where(dropoff_valid, dropoff_lon, self.dropoff_lon_median)
        df_["pickup_latitude"] = pickup_lat
        df_["pickup_longitude"] = pickup_lon
        df_["dropoff_latitude"] = dropoff_lat
        df_["dropoff_longitude"] = dropoff_lon

        AIRPORTS = {
            "JFK": (40.6413, -73.7781),
            "LGA": (40.7769, -73.8740),
            "EWR": (40.6895, -74.1745),
        }

        near_airport_features = []
        airport_near_pickup = np.zeros(len(df), dtype=np.int64)
        airport_near_dropoff = np.zeros(len(df), dtype=np.int64)

        for name, (airport_lat, airport_lon) in AIRPORTS.items():
            p_dist = haversine(
                df_["pickup_longitude"],
                airport_lon,
                df_["pickup_latitude"],
                airport_lat,
            ).to_numpy()
            d_dist = haversine(
                df_["dropoff_longitude"],
                airport_lon,
                df_["dropoff_latitude"],
                airport_lat,
            ).to_numpy()

            df_[f"pickup_dist_{name}"] = p_dist
            df_[f"dropoff_dist_{name}"] = d_dist
            near_airport_features.extend(
                [f"pickup_dist_{name}", f"dropoff_dist_{name}"]
            )

            airport_near_pickup |= (p_dist <= 2.0).astype(int)
            airport_near_dropoff |= (d_dist <= 2.0).astype(int)

        df_["airport_near_pickup"] = airport_near_pickup
        df_["airport_near_dropoff"] = airport_near_dropoff

        df_["Distance"] = haversine(
            df_["pickup_longitude"].to_numpy(),
            df_["dropoff_longitude"].to_numpy(),
            df_["pickup_latitude"].to_numpy(),
            df_["dropoff_latitude"].to_numpy(),
        )
        df_["Manhattan_Distance"] = manhattan(
            df_["pickup_longitude"].to_numpy(),
            df_["dropoff_longitude"].to_numpy(),
            df_["pickup_latitude"].to_numpy(),
            df_["dropoff_latitude"].to_numpy(),
        )
        df_["pickup_time"] = pd.to_datetime(df_["pickup_time"])

        df_.loc[df_["passenger_count"] <= 0, "passenger_count"] = 1

        df_["log_hav"] = np.log1p(np.array(df_["Distance"]))
        df_["log_manh"] = np.log1p(np.array(df_["Manhattan_Distance"]))

        df_["Year"] = df_["pickup_time"].dt.year
        df_["Day_of_Week"] = df_["pickup_time"].dt.dayofweek
        df_["Hour"] = df_["pickup_time"].dt.hour
        df_["Hour_sin"] = np.sin(2 * np.pi * df_["Hour"] / 24)

        is_weekday = df_["Day_of_Week"] < 5
        is_rush = df_["Hour"].between(7, 9) | df_["Hour"].between(16, 19)
        df_["rush"] = (is_weekday & is_rush).astype(int)
        df_["Distance_x_rush"] = df_["Distance"] * df_["rush"]
        df_["square_hav"] = df_["Distance"] ** 2
        df_["sqrt_hav"] = df_["Distance"] ** 0.5
        df_["square_manh"] = df_["Manhattan_Distance"] ** 2
        df_["sqrt_manh"] = df_["Manhattan_Distance"] ** 0.5
        df_["post_2012"] = (df_["Year"] >= 2012).astype(int)
        # 2. Stack all features into array
        feature_cols = [
            "Distance",
            "Manhattan_Distance",
            "log_hav",
            "post_2012",
            "Year",
            "rush",
            "sqrt_hav",
            "Distance_x_rush",
            "Hour_sin",
            "log_manh",
            "airport_near_pickup",
            "airport_near_dropoff",
            "square_hav",
            "square_manh",
        ] + near_airport_features
        X = df_[feature_cols].values
        X = np.asarray(X, dtype=np.float64)
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fits mean/std on training data and returns transformed feature matrix."""
        X_raw = self._extract_features(df)
        self.mean = np.mean(X_raw, axis=0)
        self.std = np.std(X_raw, axis=0) + 1e-8
        return (X_raw - self.mean) / self.std

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms validation/test data using fitted training statistics."""
        if self.mean is None or self.std is None:
            raise ValueError(
                "Pipeline is not fitted yet. Call fit_transform on training data first."
            )
        X_raw = self._extract_features(df)
        return (X_raw - self.mean) / self.std
