"""
Flight Recommendation System - Minimal Preprocessing Pipeline (Phase 1 Refactored)

Optimized for graph-based recommendations:
1. Clean essential features only
2. Build interaction matrix (user-flight graph)
3. Encode categorical features
4. Normalize numerical features
5. Save ML-ready datasets

Pipeline: Load → Clean → Encode → Normalize → Build Interactions → Save
"""

import os
import logging
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# CONFIGURATION (Inline - No external config file)
# ============================================================================

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

# Data loading
USERS_FILE = "users.csv"
FLIGHTS_FILE = "flights.csv"
BOOKINGS_FILE = "bookings.csv"

# Output files
USERS_OUT = "users_processed.csv"
FLIGHTS_OUT = "flights_processed.csv"
INTERACTIONS_OUT = "interactions.csv"

# Feature configuration
USER_FEATURES = ["age", "gender", "occupation", "travel_purpose"]
FLIGHT_FEATURES = ["price", "flight_duration_minutes", "num_layovers", "historical_delay_rate"]

# Encoding
ONE_HOT_USER_FEATURES = ["gender", "occupation", "travel_purpose"]
ONE_HOT_FLIGHT_FEATURES = []  # None for flights in minimal version

# Normalization
NORMALIZE_FEATURES = ["age", "price", "flight_duration_minutes", "historical_delay_rate"]

# Data validation
AGE_RANGE = (18, 100)
MIN_PRICE = 0

# ============================================================================
# SETUP LOGGING
# ============================================================================

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "preprocess.log")),
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
    Load raw data from CSV files.

    Args:
        users_path: Path to users.csv
        flights_path: Path to flights.csv
        bookings_path: Path to bookings.csv

    Returns:
        Tuple of (users_df, flights_df, bookings_df)
    """
    users_path = users_path or os.path.join(DATA_RAW_DIR, USERS_FILE)
    flights_path = flights_path or os.path.join(DATA_RAW_DIR, FLIGHTS_FILE)

    logger.info(f"Loading data from {DATA_RAW_DIR}")

    # Load users
    assert os.path.exists(users_path), f"Users file not found: {users_path}"
    users_df = pd.read_csv(users_path)
    logger.info(f"Loaded users: {users_df.shape[0]} rows, {users_df.shape[1]} columns")

    # Load flights
    assert os.path.exists(flights_path), f"Flights file not found: {flights_path}"
    flights_df = pd.read_csv(flights_path)
    logger.info(f"Loaded flights: {flights_df.shape[0]} rows, {flights_df.shape[1]} columns")

    # Load bookings (optional but recommended for interaction matrix)
    bookings_df = None
    if bookings_path and os.path.exists(bookings_path):
        bookings_df = pd.read_csv(bookings_path)
        logger.info(
            f"Loaded bookings: {bookings_df.shape[0]} rows, {bookings_df.shape[1]} columns"
        )
    elif os.path.exists(os.path.join(DATA_RAW_DIR, BOOKINGS_FILE)):
        bookings_df = pd.read_csv(os.path.join(DATA_RAW_DIR, BOOKINGS_FILE))
        logger.info(
            f"Loaded bookings: {bookings_df.shape[0]} rows, {bookings_df.shape[1]} columns"
        )

    return users_df, flights_df, bookings_df


# ============================================================================
# DATA CLEANING
# ============================================================================


def clean_data(
    users_df: pd.DataFrame, flights_df: pd.DataFrame, bookings_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Clean data: standardize columns, handle missing values, remove duplicates.

    Args:
        users_df: Raw users DataFrame
        flights_df: Raw flights DataFrame
        bookings_df: Raw bookings DataFrame (optional)

    Returns:
        Tuple of cleaned DataFrames
    """
    logger.info("Cleaning data...")

    # Standardize column names (lowercase, underscores)
    users_df.columns = users_df.columns.str.lower().str.replace(" ", "_")
    flights_df.columns = flights_df.columns.str.lower().str.replace(" ", "_")
    if bookings_df is not None:
        bookings_df.columns = bookings_df.columns.str.lower().str.replace(" ", "_")

    # Handle column name variations
    col_renames = {
        "user_id": "user_id",
        "flight_id": "flight_id",
        "age": "age",
        "gender": "gender",
        "occupation": "occupation",
        "travel_purpose": "travel_purpose",
        "education": "education",
        "price": "price",
        "duration": "flight_duration_minutes",
        "flight_duration": "flight_duration_minutes",
        "layovers": "num_layovers",
        "delay_rate": "historical_delay_rate",
    }
    users_df.rename(columns=col_renames, inplace=True, errors="ignore")
    flights_df.rename(columns=col_renames, inplace=True, errors="ignore")

    # Remove duplicates (log how many)
    users_before = len(users_df)
    flights_before = len(flights_df)
    users_df = users_df.drop_duplicates(subset=["user_id"])
    flights_df = flights_df.drop_duplicates(subset=["flight_id"])
    logger.info(f"Removed {users_before - len(users_df)} duplicate users")
    logger.info(f"Removed {flights_before - len(flights_df)} duplicate flights")

    # Handle missing values (simple: drop rows with critical NaNs)
    users_df = users_df.dropna(subset=["user_id", "age", "gender"])
    flights_df = flights_df.dropna(subset=["flight_id", "price"])

    # Fill remaining NaNs
    users_df["occupation"].fillna("Unknown", inplace=True)
    users_df["travel_purpose"].fillna("Leisure", inplace=True)
    flights_df["flight_duration_minutes"].fillna(flights_df["flight_duration_minutes"].median(), inplace=True)
    flights_df["num_layovers"].fillna(0, inplace=True)
    flights_df["historical_delay_rate"].fillna(0.0, inplace=True)

    # Validate ranges
    users_df = users_df[(users_df["age"] >= AGE_RANGE[0]) & (users_df["age"] <= AGE_RANGE[1])]
    flights_df = flights_df[flights_df["price"] > MIN_PRICE]

    logger.info(f"After cleaning: {len(users_df)} users, {len(flights_df)} flights")
    return users_df, flights_df, bookings_df


# ============================================================================
# INTERACTION MATRIX (Critical for graph-based recommendations)
# ============================================================================


def build_interaction_matrix(
    bookings_df: Optional[pd.DataFrame], users_df: pd.DataFrame, flights_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build user-flight interaction matrix from bookings.

    Args:
        bookings_df: Optional bookings data (user_id, flight_id, rating/satisfaction)
        users_df: Users DataFrame (for validation)
        flights_df: Flights DataFrame (for validation)

    Returns:
        DataFrame with columns: user_id, flight_id, interaction
        where interaction = 1 if booked, or rating if available
    """
    logger.info("Building interaction matrix...")

    if bookings_df is not None and len(bookings_df) > 0:
        # Rename columns if necessary
        bookings_df.columns = bookings_df.columns.str.lower().str.replace(" ", "_")

        # Create interaction (use rating if available, else 1)
        if "satisfaction_rating" in bookings_df.columns:
            interactions = bookings_df[["user_id", "flight_id", "satisfaction_rating"]].copy()
            interactions.rename(columns={"satisfaction_rating": "interaction"}, inplace=True)
        elif "rating" in bookings_df.columns:
            interactions = bookings_df[["user_id", "flight_id", "rating"]].copy()
            interactions.rename(columns={"rating": "interaction"}, inplace=True)
        else:
            interactions = bookings_df[["user_id", "flight_id"]].copy()
            interactions["interaction"] = 1

        # Validate: remove interactions for non-existent users/flights
        valid_users = set(users_df["user_id"])
        valid_flights = set(flights_df["flight_id"])
        interactions = interactions[
            (interactions["user_id"].isin(valid_users))
            & (interactions["flight_id"].isin(valid_flights))
        ]

        logger.info(f"Interaction matrix: {len(interactions)} interactions")
        return interactions
    else:
        # No bookings: return empty interaction matrix
        logger.warning("No bookings data provided. Returning empty interaction matrix.")
        return pd.DataFrame(columns=["user_id", "flight_id", "interaction"])


# ============================================================================
# FEATURE ENCODING
# ============================================================================


def encode_features(
    users_df: pd.DataFrame, flights_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode categorical features.

    Args:
        users_df: Users DataFrame
        flights_df: Flights DataFrame

    Returns:
        Tuple of encoded (users_df, flights_df)
    """
    logger.info("Encoding categorical features...")

    # One-hot encode user features
    for col in ONE_HOT_USER_FEATURES:
        if col in users_df.columns:
            dummies = pd.get_dummies(users_df[col], prefix=col, drop_first=False)
            users_df = pd.concat([users_df, dummies], axis=1)
            users_df.drop(columns=[col], inplace=True)
            logger.info(f"One-hot encoded {col}: {dummies.shape[1]} categories")

    # One-hot encode flight features (if any)
    for col in ONE_HOT_FLIGHT_FEATURES:
        if col in flights_df.columns:
            dummies = pd.get_dummies(flights_df[col], prefix=col, drop_first=False)
            flights_df = pd.concat([flights_df, dummies], axis=1)
            flights_df.drop(columns=[col], inplace=True)
            logger.info(f"One-hot encoded {col}: {dummies.shape[1]} categories")

    return users_df, flights_df


# ============================================================================
# FEATURE NORMALIZATION
# ============================================================================


def normalize_features(
    users_df: pd.DataFrame, flights_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    MinMax normalize numerical features to [0, 1].

    Args:
        users_df: Users DataFrame
        flights_df: Flights DataFrame

    Returns:
        Tuple of normalized (users_df, flights_df)
    """
    logger.info("Normalizing numerical features...")

    scaler = MinMaxScaler()

    # Normalize user features
    user_normalize_cols = [col for col in NORMALIZE_FEATURES if col in users_df.columns]
    if user_normalize_cols:
        users_df[user_normalize_cols] = scaler.fit_transform(users_df[user_normalize_cols])
        logger.info(f"Normalized {len(user_normalize_cols)} user features to [0, 1]")

    # Normalize flight features
    flight_normalize_cols = [col for col in NORMALIZE_FEATURES if col in flights_df.columns]
    if flight_normalize_cols:
        flights_df[flight_normalize_cols] = scaler.fit_transform(flights_df[flight_normalize_cols])
        logger.info(f"Normalized {len(flight_normalize_cols)} flight features to [0, 1]")

    return users_df, flights_df


# ============================================================================
# SAVE OUTPUTS
# ============================================================================


def save_datasets(
    users_df: pd.DataFrame,
    flights_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    output_dir: str = DATA_PROCESSED_DIR,
) -> None:
    """
    Save processed datasets to CSVs.

    Args:
        users_df: Processed users DataFrame
        flights_df: Processed flights DataFrame
        interactions_df: User-flight interaction matrix
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    users_path = os.path.join(output_dir, USERS_OUT)
    flights_path = os.path.join(output_dir, FLIGHTS_OUT)
    interactions_path = os.path.join(output_dir, INTERACTIONS_OUT)

    users_df.to_csv(users_path, index=False)
    flights_df.to_csv(flights_path, index=False)
    interactions_df.to_csv(interactions_path, index=False)

    logger.info(f"✅ Saved {users_path}")
    logger.info(f"✅ Saved {flights_path}")
    logger.info(f"✅ Saved {interactions_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def preprocess_pipeline(
    users_path: Optional[str] = None,
    flights_path: Optional[str] = None,
    bookings_path: Optional[str] = None,
    output_dir: str = DATA_PROCESSED_DIR,
) -> dict:
    """
    Complete preprocessing pipeline for graph-based recommendations.

    Pipeline:
    1. Load data
    2. Clean data (missing values, duplicates, validation)
    3. Encode categorical features (one-hot)
    4. Normalize numerical features (MinMax to [0,1])
    5. Build interaction matrix
    6. Save outputs

    Args:
        users_path: Path to users.csv
        flights_path: Path to flights.csv
        bookings_path: Path to bookings.csv
        output_dir: Output directory for processed files

    Returns:
        dict with status, shapes, and file paths
    """
    logger.info("=" * 80)
    logger.info("STARTING PREPROCESSING PIPELINE (MINIMAL VERSION)")
    logger.info("=" * 80)

    report = {
        "status": "initiated",
        "raw_shapes": {},
        "processed_shapes": {},
        "output_dir": output_dir,
    }

    try:
        # Step 1: Load
        logger.info("\n[1] Loading data...")
        users_df, flights_df, bookings_df = load_data(users_path, flights_path, bookings_path)
        report["raw_shapes"]["users"] = users_df.shape
        report["raw_shapes"]["flights"] = flights_df.shape

        # Step 2: Clean
        logger.info("\n[2] Cleaning data...")
        users_df, flights_df, bookings_df = clean_data(users_df, flights_df, bookings_df)

        # Step 3: Encode
        logger.info("\n[3] Encoding categorical features...")
        users_df, flights_df = encode_features(users_df, flights_df)

        # Step 4: Normalize
        logger.info("\n[4] Normalizing numerical features...")
        users_df, flights_df = normalize_features(users_df, flights_df)

        # Step 5: Build interaction matrix
        logger.info("\n[5] Building interaction matrix...")
        interactions_df = build_interaction_matrix(bookings_df, users_df, flights_df)

        # Step 6: Save
        logger.info("\n[6] Saving processed datasets...")
        save_datasets(users_df, flights_df, interactions_df, output_dir)

        # Report
        report["processed_shapes"]["users"] = users_df.shape
        report["processed_shapes"]["flights"] = flights_df.shape
        report["processed_shapes"]["interactions"] = interactions_df.shape
        report["status"] = "completed"

        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Users: {report['raw_shapes']['users']} → {report['processed_shapes']['users']}")
        logger.info(f"Flights: {report['raw_shapes']['flights']} → {report['processed_shapes']['flights']}")
        logger.info(f"Interactions: {report['processed_shapes']['interactions']}")

        return report

    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        report["status"] = "failed"
        report["error"] = str(e)
        raise


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    report = preprocess_pipeline()
    print("\n✅ PIPELINE REPORT:")
    for key, value in report.items():
        print(f"  {key}: {value}")
