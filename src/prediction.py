"""
Machine Learning Prediction Module.

Trains Random Forest regressors to predict EV charging station coordinates
(latitude, longitude) based on commissioning year and federal state.

Two modes:
    1. Model comparison: RF vs. KNN with train/test split evaluation
    2. Per-year model training: cumulative models for each year (2010-2022),
       used by the Genetic Algorithm to generate candidate station locations.

Usage:
    python -m src.prediction                   # full pipeline
    python -m src.prediction --compare-only    # model comparison only
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

from .config import CLEANED_CSV, MODELS_DIR, STATE_NAME_CORRECTIONS

logger = logging.getLogger(__name__)


def load_data(csv_path: Path = CLEANED_CSV) -> pd.DataFrame:
    """Load and prepare data for ML training."""
    data = pd.read_csv(csv_path, encoding="utf-8")
    data["year"] = pd.to_datetime(data["commissioning_date"]).dt.year
    data["federal_state"] = data["federal_state"].replace(STATE_NAME_CORRECTIONS)
    return data


def compare_models(data: pd.DataFrame) -> dict:
    """
    Train and evaluate Random Forest vs. KNN on a single train/test split.

    Returns:
        Dictionary with metric scores for both models.
    """
    subset = data[["year", "federal_state", "latitude_[dg]", "longitude_[dg]"]].copy()
    subset = pd.get_dummies(subset, columns=["federal_state"])

    X = subset.drop(["latitude_[dg]", "longitude_[dg]"], axis=1)
    y_lat = subset["latitude_[dg]"]
    y_lon = subset["longitude_[dg]"]

    X_train, X_test, y_lat_train, y_lat_test, y_lon_train, y_lon_test = (
        train_test_split(X, y_lat, y_lon, test_size=0.2, random_state=42)
    )

    results = {}
    for name, ModelClass, kwargs in [
        ("RandomForest", RandomForestRegressor, {"n_estimators": 100, "random_state": 42}),
        ("KNN", KNeighborsRegressor, {"n_neighbors": 5}),
    ]:
        model_lat = ModelClass(**kwargs).fit(X_train, y_lat_train)
        model_lon = ModelClass(**kwargs).fit(X_train, y_lon_train)
        pred_lat = model_lat.predict(X_test)
        pred_lon = model_lon.predict(X_test)

        results[name] = {
            "MAE_lat": mean_absolute_error(y_lat_test, pred_lat),
            "MAE_lon": mean_absolute_error(y_lon_test, pred_lon),
            "MSE_lat": mean_squared_error(y_lat_test, pred_lat),
            "MSE_lon": mean_squared_error(y_lon_test, pred_lon),
            "R2_lat": r2_score(y_lat_test, pred_lat),
            "R2_lon": r2_score(y_lon_test, pred_lon),
        }
        logger.info(
            "%s — R²(lat)=%.4f  R²(lon)=%.4f",
            name, results[name]["R2_lat"], results[name]["R2_lon"],
        )

    return results


def train_yearly_models(data: pd.DataFrame) -> None:
    """
    Train per-year Random Forest models (cumulative training data).

    For each year from 2010 to max_year, trains two RF models:
        - Latitude predictor
        - Longitude predictor

    Models are saved as pickle files under models/{year}/.
    """
    le = LabelEncoder()
    data = data.copy()
    data["federal_state_encoded"] = le.fit_transform(data["federal_state"])

    X = data[["year", "federal_state_encoded"]]
    y_lat = data["latitude_[dg]"]
    y_lon = data["longitude_[dg]"]
    max_year = data["year"].max()

    for year in range(2010, max_year + 1):
        mask = data["year"] <= year
        X_yr, y_lat_yr, y_lon_yr = X[mask], y_lat[mask], y_lon[mask]

        rf_lat = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_lat.fit(X_yr, y_lat_yr)
        rf_lon = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_lon.fit(X_yr, y_lon_yr)

        year_dir = MODELS_DIR / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf_lat, year_dir / f"rf_model_{year}_lat.pkl")
        joblib.dump(rf_lon, year_dir / f"rf_model_{year}_lon.pkl")

        logger.info("Year %d: trained on %d samples → %s", year, len(X_yr), year_dir)


def run(compare_only: bool = False) -> None:
    """Execute the full prediction pipeline."""
    data = load_data()

    print("=" * 60)
    print("MODEL COMPARISON: Random Forest vs. KNN")
    print("=" * 60)
    results = compare_models(data)
    for model, metrics in results.items():
        print(f"\n  {model}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    if not compare_only:
        print(f"\n{'=' * 60}")
        print("TRAINING PER-YEAR MODELS")
        print("=" * 60)
        train_yearly_models(data)
        print("\nAll models saved to", MODELS_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train location prediction models")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only compare RF vs KNN, skip per-year training")
    args = parser.parse_args()
    run(compare_only=args.compare_only)
