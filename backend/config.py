"""
Configuration and constants for the Flight Recommendation System.
Centralized settings for preprocessing, feature engineering, and ML pipelines.
"""

import os
from enum import Enum

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ============================================================================
# DATA LOADING CONFIG
# ============================================================================

# Expected file names in data/raw/
USERS_FILE = "users.csv"
FLIGHTS_FILE = "flights.csv"
BOOKINGS_FILE = "bookings.csv"  # Optional: user-flight historical interactions

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# Budget sensitivity: price range thresholds (in USD)
BUDGET_BRACKETS = {
    "ultra_budget": 50,
    "budget": 150,
    "economy": 300,
    "premium": 600,
    "luxury": float("inf"),
}

# Loyalty tiers: based on cumulative bookings
LOYALTY_TIERS = {
    "bronze": 0,
    "silver": 5,
    "gold": 15,
    "platinum": 30,
}

# Seasonality: typical high-travel periods (month numbers)
PEAK_MONTHS = [6, 7, 8, 12]  # Summer + December holidays
OFF_PEAK_MONTHS = [2, 9, 10, 11]  # Low-season months

# Route popularity thresholds
POPULAR_ROUTE_MIN_FLIGHTS = 50  # Min flights per month to be "popular"

# ============================================================================
# NORMALIZATION CONFIG
# ============================================================================

# Features to normalize (0-1 scaling)
NUMERICAL_FEATURES_TO_NORMALIZE = [
    "age",
    "price",
    "booking_lead_time",
    "flight_duration_minutes",
    "num_layovers",
    "airline_rating",
    "historical_delay_rate",
    "weather_temperature",
    "weather_wind_speed",
    "weather_visibility",
]

# Features to log-transform before normalization (to handle skewness)
LOG_TRANSFORM_FEATURES = [
    "price",
    "booking_lead_time",
    "airline_rating",
]

# ============================================================================
# CATEGORICAL FEATURES CONFIG
# ============================================================================

# One-hot encoding features
ONE_HOT_FEATURES = [
    "gender",
    "seat_class",
    "airline",
    "travel_purpose",
]

# Ordinal features (order matters, map to integer)
ORDINAL_FEATURES = {
    "education": {
        "high_school": 1,
        "bachelors": 2,
        "masters": 3,
        "phd": 4,
    },
    "seat_class": {
        "economy": 1,
        "premium_economy": 2,
        "business": 3,
        "first": 4,
    },
}

# ============================================================================
# MISSING VALUE STRATEGY
# ============================================================================

MISSING_VALUE_STRATEGY = {
    # Numerical features: use median
    "age": "median",
    "price": "median",
    "booking_lead_time": "median",
    "flight_duration_minutes": "median",
    "num_layovers": "median",
    "airline_rating": "median",
    "historical_delay_rate": "median",
    "weather_temperature": "median",
    "weather_wind_speed": "median",
    "weather_visibility": "median",
    # Categorical: use mode (most frequent)
    "gender": "mode",
    "occupation": "mode",
    "education": "mode",
    "seat_class": "mode",
    "travel_purpose": "mode",
    "airline": "mode",
    # Special: forward-fill or drop
    "loyalty_score": "fill_zero",  # No bookings = 0 loyalty
    "travel_history": "drop_row",  # Drop rows with missing travel history
}

# ============================================================================
# OUTPUT CONFIG
# ============================================================================

# Output file names
PROCESSED_USERS_FILE = "users_processed.csv"
PROCESSED_FLIGHTS_FILE = "flights_processed.csv"
PROCESSED_BOOKINGS_FILE = "bookings_processed.csv"

# Feature scaling metadata (saved for inference)
SCALING_METADATA_FILE = "scaling_metadata.json"
ENCODING_METADATA_FILE = "encoding_metadata.json"

# ============================================================================
# LOGGING CONFIG
# ============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# ============================================================================
# COLUMN RENAMES (for consistency)
# ============================================================================

# Standardize column names across raw datasets
COLUMN_RENAME_MAP = {
    "User_ID": "user_id",
    "user_id": "user_id",
    "Age": "age",
    "age": "age",
    "Gender": "gender",
    "gender": "gender",
    "Occupation": "occupation",
    "occupation": "occupation",
    "Education": "education",
    "education": "education",
    "TravelPurpose": "travel_purpose",
    "travel_purpose": "travel_purpose",
    "PreferredAirline": "preferred_airline",
    "preferred_airline": "preferred_airline",
    "Flight_ID": "flight_id",
    "flight_id": "flight_id",
    "Airline": "airline",
    "airline": "airline",
    "Price": "price",
    "price": "price",
    "Duration": "flight_duration_minutes",
    "flight_duration_minutes": "flight_duration_minutes",
    "Layovers": "num_layovers",
    "num_layovers": "num_layovers",
    "AirlineRating": "airline_rating",
    "airline_rating": "airline_rating",
    "SeatClass": "seat_class",
    "seat_class": "seat_class",
    "DelayRate": "historical_delay_rate",
    "historical_delay_rate": "historical_delay_rate",
}
