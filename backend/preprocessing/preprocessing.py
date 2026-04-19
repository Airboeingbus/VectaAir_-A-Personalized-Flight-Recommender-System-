"""
Flight Recommendation System - Preprocessing Module (Phase 1)

This module handles data cleaning, normalization, encoding, and feature engineering
for the Flight Recommendation System pipeline.

Main functions:
- load_data(): Load raw user and flight datasets
- clean_data(): Handle missing values and data validation
- normalize_features(): Normalize numerical features
- encode_features(): Encode categorical features
- engineer_features(): Create new derived features
- preprocess_pipeline(): Complete preprocessing pipeline
"""

import logging
import os
import json
from typing import Tuple, Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

from config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    LOGS_DIR,
    USERS_FILE,
    FLIGHTS_FILE,
    BOOKINGS_FILE,
    COLUMN_RENAME_MAP,
    MISSING_VALUE_STRATEGY,
    NUMERICAL_FEATURES_TO_NORMALIZE,
    LOG_TRANSFORM_FEATURES,
    ONE_HOT_FEATURES,
    ORDINAL_FEATURES,
    BUDGET_BRACKETS,
    LOYALTY_TIERS,
    PEAK_MONTHS,
    OFF_PEAK_MONTHS,
    POPULAR_ROUTE_MIN_FLIGHTS,
    PROCESSED_USERS_FILE,
    PROCESSED_FLIGHTS_FILE,
    PROCESSED_BOOKINGS_FILE,
    SCALING_METADATA_FILE,
    ENCODING_METADATA_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
)

warnings.filterwarnings("ignore")

# ============================================================================
# LOGGING SETUP
# ============================================================================

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "preprocessing.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================


def load_data(
    users_path: Optional[str] = None,
    flights_path: Optional[str] = None,
    bookings_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load raw user, flight, and optional booking data from CSV files.

    Args:
        users_path: Path to users CSV. Defaults to DATA_RAW_DIR/USERS_FILE.
        flights_path: Path to flights CSV. Defaults to DATA_RAW_DIR/FLIGHTS_FILE.
        bookings_path: Path to bookings CSV (optional).

    Returns:
        Tuple of (users_df, flights_df, bookings_df or None)

    Raises:
        FileNotFoundError: If required files don't exist.
        pd.errors.ParserError: If CSV parsing fails.
    """
    users_path = users_path or os.path.join(DATA_RAW_DIR, USERS_FILE)
    flights_path = flights_path or os.path.join(DATA_RAW_DIR, FLIGHTS_FILE)

    logger.info(f"Loading data from {DATA_RAW_DIR}")

    # Load users
    if not os.path.exists(users_path):
        raise FileNotFoundError(f"Users file not found: {users_path}")
    users_df = pd.read_csv(users_path)
    logger.info(f"Loaded users: {users_df.shape[0]} rows, {users_df.shape[1]} columns")

    # Load flights
    if not os.path.exists(flights_path):
        raise FileNotFoundError(f"Flights file not found: {flights_path}")
    flights_df = pd.read_csv(flights_path)
    logger.info(f"Loaded flights: {flights_df.shape[0]} rows, {flights_df.shape[1]} columns")

    # Load bookings (optional)
    bookings_df = None
    if bookings_path:
        if os.path.exists(bookings_path):
            bookings_df = pd.read_csv(bookings_path)
            logger.info(
                f"Loaded bookings: {bookings_df.shape[0]} rows, {bookings_df.shape[1]} columns"
            )
    elif os.path.exists(os.path.join(DATA_RAW_DIR, BOOKINGS_FILE)):
        bookings_path = os.path.join(DATA_RAW_DIR, BOOKINGS_FILE)
        bookings_df = pd.read_csv(bookings_path)
        logger.info(
            f"Loaded bookings: {bookings_df.shape[0]} rows, {bookings_df.shape[1]} columns"
        )

    return users_df, flights_df, bookings_df


# ============================================================================
# DATA CLEANING
# ============================================================================


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names using predefined mapping.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True, errors="ignore")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    logger.info(f"Standardized columns to: {list(df.columns)}")
    return df


def handle_missing_values(df: pd.DataFrame, data_type: str = "users") -> pd.DataFrame:
    """
    Handle missing values using predefined strategies (median, mode, fill, drop).

    Args:
        df: Input DataFrame
        data_type: Type of data ('users', 'flights', 'bookings') for context logging

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    initial_nulls = df.isnull().sum().sum()

    logger.info(f"Handling missing values for {data_type}...")

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue  # Skip columns with no missing values

        strategy = MISSING_VALUE_STRATEGY.get(col, "median")

        if strategy == "median":
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.debug(f"  {col}: Filled {df[col].isnull().sum()} NaNs with median={median_val}")

        elif strategy == "mode":
            mode_val = df[col].mode()[0] if not df[col].mode().empty else df[col].iloc[0]
            df[col].fillna(mode_val, inplace=True)
            logger.debug(f"  {col}: Filled {df[col].isnull().sum()} NaNs with mode={mode_val}")

        elif strategy == "fill_zero":
            df[col].fillna(0, inplace=True)
            logger.debug(f"  {col}: Filled {df[col].isnull().sum()} NaNs with 0")

        elif strategy == "drop_row":
            before = len(df)
            df = df.dropna(subset=[col])
            dropped = before - len(df)
            logger.debug(f"  {col}: Dropped {dropped} rows")

    final_nulls = df.isnull().sum().sum()
    logger.info(f"Missing values: {initial_nulls} → {final_nulls}")
    return df


def validate_data(df: pd.DataFrame, data_type: str = "users") -> Tuple[pd.DataFrame, Dict]:
    """
    Validate data types and value ranges, log issues.

    Args:
        df: Input DataFrame
        data_type: Type of data for logging context

    Returns:
        Tuple of (possibly cleaned DataFrame, validation report dict)
    """
    df = df.copy()
    report = {
        "data_type": data_type,
        "rows": len(df),
        "columns": len(df.columns),
        "issues": [],
    }

    logger.info(f"Validating {data_type} data...")

    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        report["issues"].append(f"Found {duplicates} duplicate rows")
        logger.warning(f"Found {duplicates} duplicate rows in {data_type}")
        df = df.drop_duplicates()

    # Age range validation (users)
    if "age" in df.columns:
        invalid_age = ((df["age"] < 18) | (df["age"] > 100)).sum()
        if invalid_age > 0:
            report["issues"].append(f"Found {invalid_age} invalid ages (not in 18-100)")
            logger.warning(f"Found {invalid_age} invalid ages in {data_type}")
            df = df[(df["age"] >= 18) & (df["age"] <= 100)]

    # Price validation (flights)
    if "price" in df.columns:
        invalid_price = (df["price"] <= 0).sum()
        if invalid_price > 0:
            report["issues"].append(f"Found {invalid_price} non-positive prices")
            logger.warning(f"Found {invalid_price} non-positive prices")
            df = df[df["price"] > 0]

    logger.info(f"Validation complete. Rows after cleaning: {len(df)}")
    return df, report


# ============================================================================
# FEATURE NORMALIZATION
# ============================================================================


def normalize_numerical_features(
    df: pd.DataFrame, metadata_save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize numerical features using MinMaxScaler (0-1 range).
    Applies log transformation to skewed features first.

    Args:
        df: Input DataFrame with numerical features
        metadata_save_path: Path to save scaling metadata (min/max values)

    Returns:
        Tuple of (normalized DataFrame, scaling metadata dict)
    """
    df = df.copy()
    metadata = {"scaler_type": "minmax", "features": {}}

    logger.info("Normalizing numerical features...")

    # Identify numerical columns to normalize
    cols_to_normalize = [col for col in NUMERICAL_FEATURES_TO_NORMALIZE if col in df.columns]

    for col in cols_to_normalize:
        if df[col].isnull().sum() > 0:
            logger.warning(f"Column {col} has missing values. Skipping normalization.")
            continue

        # Apply log transformation if needed (add 1 to avoid log(0))
        if col in LOG_TRANSFORM_FEATURES:
            df_col = np.log1p(df[col].values)
            logger.debug(f"Applied log transformation to {col}")
        else:
            df_col = df[col].values

        # MinMax scaling (0-1)
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_col.reshape(-1, 1)).flatten()

        # Store metadata
        metadata["features"][col] = {
            "min": float(scaler.data_min_[0]),
            "max": float(scaler.data_max_[0]),
            "log_transformed": col in LOG_TRANSFORM_FEATURES,
        }

        df[col] = df_scaled
        logger.debug(f"{col}: Normalized to [0, 1]")

    # Save metadata if path provided
    if metadata_save_path:
        os.makedirs(os.path.dirname(metadata_save_path) or ".", exist_ok=True)
        with open(metadata_save_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved scaling metadata to {metadata_save_path}")

    return df, metadata


# ============================================================================
# FEATURE ENCODING
# ============================================================================


def encode_categorical_features(
    df: pd.DataFrame, metadata_save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical features using one-hot encoding.
    Maintains mapping for inference.

    Args:
        df: Input DataFrame
        metadata_save_path: Path to save encoding metadata (mappings)

    Returns:
        Tuple of (encoded DataFrame, encoding metadata dict)
    """
    df = df.copy()
    metadata = {"encoding_type": "one_hot", "features": {}}

    logger.info("Encoding categorical features...")

    for col in ONE_HOT_FEATURES:
        if col not in df.columns:
            logger.debug(f"Column {col} not found in DataFrame. Skipping.")
            continue

        # Get unique values (categories)
        categories = sorted(df[col].unique().astype(str))
        metadata["features"][col] = {"categories": categories}

        # One-hot encode
        one_hot = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, one_hot], axis=1)

        logger.debug(f"{col}: One-hot encoded into {len(categories)} categories")
        logger.debug(f"  Categories: {categories}")

    # Drop original categorical columns (after encoding)
    cols_to_drop = [col for col in ONE_HOT_FEATURES if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Save metadata if path provided
    if metadata_save_path:
        os.makedirs(os.path.dirname(metadata_save_path) or ".", exist_ok=True)
        with open(metadata_save_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved encoding metadata to {metadata_save_path}")

    return df, metadata


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode ordinal categorical features (order matters).

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with ordinal features encoded numerically
    """
    df = df.copy()
    logger.info("Encoding ordinal features...")

    for feature, mapping in ORDINAL_FEATURES.items():
        if feature in df.columns:
            df[feature] = df[feature].map(mapping)
            logger.debug(f"{feature}: Encoded with mapping {mapping}")

    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


def engineer_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer derived features for user data:
    - Budget sensitivity: category based on typical booking price
    - Loyalty score: combined metric from tier and booking frequency

    Args:
        df: Users DataFrame

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    logger.info("Engineering user features...")

    # Budget sensitivity: Map average spending to budget brackets
    if "price" in df.columns:
        df["budget_sensitivity"] = pd.cut(
            df["price"],
            bins=[0, 50, 150, 300, 600, float("inf")],
            labels=["ultra_budget", "budget", "economy", "premium", "luxury"],
            include_lowest=True,
        )
        logger.debug("Created budget_sensitivity feature")

    # Loyalty score: 0-100 scale
    # Simplified: based on booking frequency if available
    if "num_bookings" in df.columns:
        df["loyalty_score"] = np.minimum(df["num_bookings"] * 10, 100)
        logger.debug("Created loyalty_score based on num_bookings")
    else:
        df["loyalty_score"] = 50  # Default middle value
        logger.debug("Set default loyalty_score = 50")

    return df


def engineer_flight_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer derived features for flight data:
    - Price volatility: coefficient of variation of prices on same route
    - Route popularity: normalized by total flights
    - Effective delay penalty: delay_rate weighted by flight_duration
    - Seasonality factor: 1 for peak, 0.5 for off-peak months

    Args:
        df: Flights DataFrame

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    logger.info("Engineering flight features...")

    # Route popularity (simplified to airline + destination representation)
    if "airline" in df.columns and "destination" in df.columns:
        route_counts = df.groupby(["airline", "destination"]).size()
        df["route_popularity"] = df.apply(
            lambda row: min(route_counts.get((row["airline"], row["destination"]), 0) / 100, 1.0),
            axis=1,
        )
        logger.debug("Created route_popularity feature")
    else:
        df["route_popularity"] = 0.5
        logger.debug("Set default route_popularity = 0.5")

    # Effective delay penalty: combine delay rate and flight duration
    if "historical_delay_rate" in df.columns and "flight_duration_minutes" in df.columns:
        df["delay_penalty"] = (
            df["historical_delay_rate"] * (df["flight_duration_minutes"] / 600)
        )  # Normalize by ~10 hours
        logger.debug("Created delay_penalty feature")

    # Seasonality factor (simplified - based on dummy month if available)
    if "departure_month" in df.columns:
        df["seasonality_factor"] = df["departure_month"].apply(
            lambda m: 1.0 if m in PEAK_MONTHS else (0.5 if m in OFF_PEAK_MONTHS else 0.75)
        )
        logger.debug("Created seasonality_factor based on departure_month")
    else:
        df["seasonality_factor"] = 0.75
        logger.debug("Set default seasonality_factor = 0.75")

    return df


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================


def preprocess_pipeline(
    users_path: Optional[str] = None,
    flights_path: Optional[str] = None,
    bookings_path: Optional[str] = None,
    output_dir: str = DATA_PROCESSED_DIR,
) -> Dict:
    """
    Complete preprocessing pipeline: load → clean → normalize → encode → engineer features.

    Args:
        users_path: Path to users CSV
        flights_path: Path to flights CSV
        bookings_path: Path to bookings CSV
        output_dir: Directory to save processed data

    Returns:
        Dictionary with preprocessing report (shapes, features, metadata paths)
    """
    logger.info("=" * 80)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 80)

    report = {
        "status": "initiated",
        "raw_shapes": {},
        "processed_shapes": {},
        "features": {},
        "metadata_paths": {},
    }

    try:
        # 1. Load data
        logger.info("\n[STEP 1/6] Loading data...")
        users_df, flights_df, bookings_df = load_data(users_path, flights_path, bookings_path)
        report["raw_shapes"]["users"] = users_df.shape
        report["raw_shapes"]["flights"] = flights_df.shape
        if bookings_df is not None:
            report["raw_shapes"]["bookings"] = bookings_df.shape

        # 2. Standardize columns
        logger.info("\n[STEP 2/6] Standardizing columns...")
        users_df = standardize_columns(users_df)
        flights_df = standardize_columns(flights_df)
        if bookings_df is not None:
            bookings_df = standardize_columns(bookings_df)

        # 3. Clean data
        logger.info("\n[STEP 3/6] Cleaning data...")
        users_df = handle_missing_values(users_df, "users")
        flights_df = handle_missing_values(flights_df, "flights")
        if bookings_df is not None:
            bookings_df = handle_missing_values(bookings_df, "bookings")

        users_df, users_validation = validate_data(users_df, "users")
        flights_df, flights_validation = validate_data(flights_df, "flights")
        if bookings_df is not None:
            bookings_df, bookings_validation = validate_data(bookings_df, "bookings")

        report["validation"] = {
            "users": users_validation,
            "flights": flights_validation,
        }

        # 4. Engineer features
        logger.info("\n[STEP 4/6] Engineering features...")
        users_df = engineer_user_features(users_df)
        flights_df = engineer_flight_features(flights_df)

        # 5. Encode categorical features
        logger.info("\n[STEP 5/6] Encoding categorical features...")
        os.makedirs(output_dir, exist_ok=True)

        encoding_path_users = os.path.join(output_dir, "encoding_metadata_users.json")
        users_df, users_encoding = encode_categorical_features(users_df, encoding_path_users)
        report["metadata_paths"]["users_encoding"] = encoding_path_users

        encoding_path_flights = os.path.join(output_dir, "encoding_metadata_flights.json")
        flights_df, flights_encoding = encode_categorical_features(
            flights_df, encoding_path_flights
        )
        report["metadata_paths"]["flights_encoding"] = encoding_path_flights

        # Ordinal encoding
        users_df = encode_ordinal_features(users_df)
        flights_df = encode_ordinal_features(flights_df)

        # 6. Normalize numerical features
        logger.info("\n[STEP 6/6] Normalizing numerical features...")
        scaling_path_users = os.path.join(output_dir, "scaling_metadata_users.json")
        users_df, users_scaling = normalize_numerical_features(users_df, scaling_path_users)
        report["metadata_paths"]["users_scaling"] = scaling_path_users

        scaling_path_flights = os.path.join(output_dir, "scaling_metadata_flights.json")
        flights_df, flights_scaling = normalize_numerical_features(
            flights_df, scaling_path_flights
        )
        report["metadata_paths"]["flights_scaling"] = scaling_path_flights

        # Save processed data
        logger.info("\n[SAVING] Writing processed datasets...")
        users_output = os.path.join(output_dir, PROCESSED_USERS_FILE)
        flights_output = os.path.join(output_dir, PROCESSED_FLIGHTS_FILE)

        users_df.to_csv(users_output, index=False)
        flights_df.to_csv(flights_output, index=False)
        logger.info(f"Saved processed users to {users_output}")
        logger.info(f"Saved processed flights to {flights_output}")

        if bookings_df is not None:
            bookings_output = os.path.join(output_dir, PROCESSED_BOOKINGS_FILE)
            bookings_df.to_csv(bookings_output, index=False)
            logger.info(f"Saved processed bookings to {bookings_output}")

        # Report
        report["processed_shapes"]["users"] = users_df.shape
        report["processed_shapes"]["flights"] = flights_df.shape
        if bookings_df is not None:
            report["processed_shapes"]["bookings"] = bookings_df.shape

        report["features"]["users"] = list(users_df.columns)
        report["features"]["flights"] = list(flights_df.columns)

        report["status"] = "completed"

        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Users: {report['raw_shapes']['users']} → {report['processed_shapes']['users']}")
        logger.info(
            f"Flights: {report['raw_shapes']['flights']} → {report['processed_shapes']['flights']}"
        )

        return report

    except Exception as e:
        logger.error(f"ERROR in preprocessing pipeline: {str(e)}", exc_info=True)
        report["status"] = "failed"
        report["error"] = str(e)
        raise


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    """
    Run preprocessing pipeline from command line.

    Usage:
        python preprocessing.py
    """
    report = preprocess_pipeline()
    print("\n" + "=" * 80)
    print("PREPROCESSING REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2, default=str))
