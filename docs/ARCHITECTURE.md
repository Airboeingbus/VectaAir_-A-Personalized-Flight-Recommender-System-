"""
Detailed Architecture and Design Document for the Flight Recommendation System

This document explains the system design, modules, and data flow for Phase 1.
"""

# ============================================================================
# SYSTEM ARCHITECTURE OVERVIEW
# ============================================================================

"""
Flight Recommendation System - Complete Pipeline

┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAW DATA INPUT                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   users.csv      │  │   flights.csv    │  │  bookings.csv    │          │
│  │                  │  │                  │  │   (optional)     │          │
│  │ - user profiles  │  │ - flight details │  │ - interactions   │          │
│  │ - demographics   │  │ - prices         │  │ - satisfaction   │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                    │                     │
│           └─────────────────────┼────────────────────┘                     │
│                                 │                                          │
│                       [PHASE 1: PREPROCESSING]                             │
│                                 │                                          │
└─────────────────────────────────┼──────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA CLEANING                                         │
│ • Standardize column names                                                  │
│ • Handle missing values (imputation)                                        │
│ • Remove duplicates                                                         │
│ • Validate data ranges (age 18-100, price > 0, etc.)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                                     │
│                                                                             │
│  USER FEATURES:                  FLIGHT FEATURES:                          │
│  • budget_sensitivity (5 levels) • route_popularity (normalized)           │
│  • loyalty_score (0-100)         • delay_penalty (delay × duration)        │
│                                   • seasonality_factor (peak/off-peak)     │
│                                                                             │
│  ENVIRONMENTAL (future):                                                   │
│  • weather conditions, airport congestion, fuel prices                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENCODING                                        │
│ • One-hot encode: gender (inclusive), airline, seat_class, travel_purpose  │
│ • Ordinal encode: education, seat_class (if applicable)                    │
│ → Categorical → Numerical (ML-ready)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEATURE NORMALIZATION                                     │
│ • MinMax scaling: All numerical features → [0, 1]                          │
│ • Log transform: Skewed features (price, ratings) → normalize              │
│ • Save metadata: Min/max values for inference pipeline                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PROCESSED DATA OUTPUT                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │ users_processed  │  │ flights_processed│  │ Metadata JSON    │          │
│  │      .csv        │  │      .csv        │  │ (scalers/encoders)          │
│  │                  │  │                  │  │                  │          │
│  │ Rows: N_users    │  │ Rows: N_flights  │  │ encoding_users   │          │
│  │ Cols: M_features │  │ Cols: K_features │  │ scaling_users    │          │
│  │ All [0, 1]       │  │ All [0, 1]       │  │ ...              │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ▼
                  [PHASE 2: CLUSTERING & SEGMENTATION]
                  [PHASE 3: LINK PREDICTION]
                  [PHASE 4: DELAY PREDICTION (CNN)]
                  [PHASE 5: HYBRID SCORING]
                  [PHASE 6: FLASK API]
"""

# ============================================================================
# MODULE STRUCTURE (PHASE 1)
# ============================================================================

"""
Flight_Recomendor/
│
├── src/
│   ├── __init__.py              # Package marker
│   ├── config.py                # Configuration & constants
│   └── preprocessing.py         # Main preprocessing module
│
├── data/
│   ├── raw/                     # Raw input datasets
│   │   ├── users.csv            # User profiles
│   │   ├── flights.csv          # Flight listings
│   │   └── bookings.csv         # (Optional) historical bookings
│   │
│   └── processed/               # Output datasets & metadata
│       ├── users_processed.csv
│       ├── flights_processed.csv
│       ├── bookings_processed.csv
│       ├── encoding_metadata_users.json
│       ├── encoding_metadata_flights.json
│       ├── scaling_metadata_users.json
│       └── scaling_metadata_flights.json
│
├── logs/
│   ├── preprocessing.log        # Execution logs
│   └── ...
│
├── generate_sample_data.py      # Synthetic data generator (for testing)
├── examples.py                  # Example usage scripts
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation


KEY DESIGN DECISIONS:
======================

1. **Modular Functions**
   - Each function = one responsibility (SRP)
   - Composable: can run step-by-step or via pipeline
   - Easy to unit test and debug

2. **Configuration Management**
   - All constants in config.py (not hardcoded)
   - Easy to adjust parameters without changing code
   - Budget brackets, loyalty tiers, seasonality rules, etc.

3. **Metadata Persistence**
   - Scaling metadata (min/max values) saved as JSON
   - Encoding metadata (category mappings) saved as JSON
   - Enables reproducible inference on new data
   - No fitted sklearn objects (for flexibility)

4. **Logging**
   - All operations logged to file + console
   - Helps debug data quality issues
   - Tracks which rows were dropped, how NaNs were imputed, etc.

5. **Inclusive Design**
   - Gender: one-hot encoded (not binary assumption)
   - Occupation: treated as feature, not stereotype
   - Budget sensitivity: derived from DATA, not assumptions
   - No hardcoded rules about user preferences

6. **Performance**
   - Vectorized operations (pandas/numpy)
   - No loops where possible
   - Efficient memory usage (handle large datasets)

"""

# ============================================================================
# DATA FLOW DIAGRAM
# ============================================================================

"""
PHASE 1: PREPROCESSING - DETAILED DATA FLOW

INPUT: Raw CSV files
└─
  [1] LOAD
    • read_csv(users.csv) → pandas DataFrame
    • read_csv(flights.csv) → pandas DataFrame
    • read_csv(bookings.csv) → pandas DataFrame (optional)
  └─
    [2] STANDARDIZE COLUMNS
      • Rename: "Age" → "age", "Flight_ID" → "flight_id"
      • Lowercase all columns
      • Use COLUMN_RENAME_MAP from config
    └─
      [3] HANDLE MISSING VALUES
        For each column:
          • Numerical (age, price): median imputation
          • Categorical (gender, airline): mode imputation
          • Special (loyalty_score): fill with 0
          • Missing row (travel_history): drop row
      └─
        [4] VALIDATE DATA
          • Remove duplicate rows
          • Filter age: 18-100
          • Filter price: > 0
          • Log removed rows
      └─
        [5] ENGINEER FEATURES
          
          USER FEATURES:
          • budget_sensitivity: pd.cut(price) → [ultra_budget, budget, economy, premium, luxury]
          • loyalty_score: num_bookings * 10, max 100
          
          FLIGHT FEATURES:
          • route_popularity: groupby([airline, destination]).size() / 100, capped at 1.0
          • delay_penalty: historical_delay_rate * (duration / 600)
          • seasonality_factor: 1.0 (peak) / 0.75 (normal) / 0.5 (off-peak)
      └─
        [6] ENCODE CATEGORICAL
          
          ONE-HOT ENCODING:
            gender: [M, F, NB, ...] → [gender_M, gender_F, gender_NB, ...]
            airline: [United, Delta, ...] → [airline_United, airline_Delta, ...]
            seat_class: [Economy, Business, ...] → [seat_class_Economy, ...]
            travel_purpose: [Business, Leisure, ...] → [travel_purpose_Business, ...]
          
          ORDINAL ENCODING:
            education: {HS:1, Bach:2, Mast:3, PhD:4}
            seat_class (secondary): {Econ:1, Prem:2, Bus:3, First:4}
          
          SAVE: encoding_metadata.json
            {
              "features": {
                "gender": {"categories": ["Male", "Female", "Non-binary", ...]},
                "airline": {"categories": ["United", "Delta", ...]},
                ...
              }
            }
      └─
        [7] NORMALIZE NUMERICAL
          
          For each numerical feature in NUMERICAL_FEATURES_TO_NORMALIZE:
            1. Check if log-skewed (in LOG_TRANSFORM_FEATURES)
            2. If yes: apply log1p(x) = log(1 + x)
            3. Apply MinMaxScaler: (x - min) / (max - min) → [0, 1]
            4. Save metadata: min, max, log_transformed flag
          
          SAVE: scaling_metadata.json
            {
              "features": {
                "price": {
                  "min": 10.5,
                  "max": 1250.3,
                  "log_transformed": true
                },
                "age": {
                  "min": 18,
                  "max": 75,
                  "log_transformed": false
                },
                ...
              }
            }
      └─
        [8] OUTPUT
          • users_processed.csv (N × M matrix, all values ∈ [0, 1])
          • flights_processed.csv (N_flights × K matrix, all values ∈ [0, 1])
          • encoding_metadata_users.json
          • encoding_metadata_flights.json
          • scaling_metadata_users.json
          • scaling_metadata_flights.json
          • preprocessing.log (detailed execution trace)
"""

# ============================================================================
# CONFIGURATION STRUCTURE
# ============================================================================

"""
config.py - All hardcoded parameters documented here:

1. PATHS
   • PROJECT_ROOT: Calculated from __file__
   • DATA_RAW_DIR: "project_root/data/raw"
   • DATA_PROCESSED_DIR: "project_root/data/processed"
   • LOGS_DIR: "project_root/logs"

2. FEATURE ENGINEERING
   • BUDGET_BRACKETS: Price thresholds for sensitivity categories
   • LOYALTY_TIERS: Booking count thresholds for loyalty levels
   • PEAK_MONTHS: Months 6, 7, 8, 12 (high travel)
   • OFF_PEAK_MONTHS: Months 2, 9, 10, 11 (low travel)
   • POPULAR_ROUTE_MIN_FLIGHTS: >= 50 flights/month = popular

3. NORMALIZATION
   • NUMERICAL_FEATURES_TO_NORMALIZE: List of columns to normalize
   • LOG_TRANSFORM_FEATURES: Skewed features (price, ratings, etc.)

4. ENCODING
   • ONE_HOT_FEATURES: gender, airline, seat_class, travel_purpose
   • ORDINAL_FEATURES: education, seat_class (with mapping)

5. MISSING VALUE HANDLING
   • MISSING_VALUE_STRATEGY: Dict mapping column → strategy
     - "median": Use median for numerical
     - "mode": Use mode for categorical
     - "fill_zero": Fill with 0
     - "drop_row": Drop rows with this column NaN

6. OUTPUT FILES
   • PROCESSED_USERS_FILE: "users_processed.csv"
   • PROCESSED_FLIGHTS_FILE: "flights_processed.csv"
   • SCALING_METADATA_FILE: "scaling_metadata.json"
   • ENCODING_METADATA_FILE: "encoding_metadata.json"

RATIONALE FOR THESE CHOICES:
=============================

Why MinMax Scaling?
  - Preserves the shape of the original distribution
  - Bounds output to [0, 1] (good for neural networks)
  - Interpretable (0 = minimum value, 1 = maximum value)

Why Log Transform Skewed Features?
  - Price distribution is log-normal (right-skewed)
  - Log transform makes it more Gaussian
  - Better for downstream ML models

Why One-Hot for Gender?
  - INCLUSIVE: Not assuming binary gender
  - Captures multiple categories naturally
  - One-hot avoids ordinal assumption

Why Json for Metadata (not pickle)?
  - Human-readable
  - Language-agnostic (can use in Java/JS backends)
  - Safe (no arbitrary code execution risk)
  - Version control friendly

Why Separate Metadata Files?
  - Modularity: Can update encoding without touching scaling
  - Auditability: Can inspect what transformations were applied
  - Reproducibility: Can apply exact same transforms to new data
"""

# ============================================================================
# CODE QUALITY & BEST PRACTICES
# ============================================================================

"""
DESIGN PATTERNS USED:

1. **Pipeline Pattern**
   - preprocess_pipeline() orchestrates multiple steps
   - Each step is independent (can be reused)
   - Easy to add/remove steps

2. **Configuration Pattern**
   - All constants in config.py
   - No magic numbers in preprocessing.py
   - Easy to adjust behavior

3. **Logging Pattern**
   - Multiple handlers: file + console
   - Consistent format: timestamp, level, message
   - Helps debugging and monitoring

4. **Metadata Pattern**
   - Store transformation parameters as JSON
   - Enable reproducible inference
   - Audit trail of what was done

5. **Error Handling Pattern**
   - Try-catch at top level (preprocess_pipeline)
   - Validate inputs early (standardize_columns, validate_data)
   - Informative error messages

PERFORMANCE CONSIDERATIONS:

• Vectorized Operations
  ✓ Use pandas/numpy operations (not loops)
  ✓ Example: df[col].median() instead of for-loop
  
• Memory Efficiency
  ✓ Work with chunks for very large datasets
  ✓ Drop unnecessary columns after use
  ✓ Use appropriate dtypes (int vs float)

• Scalability
  ✓ Current: ~10k users, ~5k flights (seconds)
  ✓ With optimization: ~1M rows (minutes)
  ✓ For bigger data: Spark/Dask (Phase 6+)

TYPE HINTS & DOCUMENTATION:

• All functions have type hints (args and return types)
• Docstrings explain purpose, args, returns, exceptions
• Examples in docstrings for key functions
• Inline comments for non-obvious logic

TESTING STRATEGY:

generate_sample_data.py
  → Creates synthetic data matching expected schema
  
examples.py
  → Demonstrates usage patterns
  → Step-by-step example
  → Full pipeline example
  → Data loading example
  
Unit tests (future):
  → Test each function independently
  → Test error cases (missing values, outliers, etc.)
  → Test metadata persistence
"""

# ============================================================================
# PHASE 1 COMPLETION CHECKLIST
# ============================================================================

"""
✅ COMPLETED:

Data Loading:
✅ Load from CSV
✅ Handle missing input files
✅ Load optional booking data

Data Cleaning:
✅ Standardize column names
✅ Handle missing values (smart imputation)
✅ Remove duplicates
✅ Validate data ranges
✅ Log all removed/modified rows

Feature Engineering:
✅ Budget sensitivity (5 categories)
✅ Loyalty score (0-100 scale)
✅ Route popularity (normalized)
✅ Delay penalty (weighted)
✅ Seasonality factor (peak/off-peak)

Feature Encoding:
✅ One-hot encoding for categorical
✅ Ordinal encoding where applicable
✅ Save encoding metadata

Feature Normalization:
✅ MinMax scaling to [0, 1]
✅ Log transformation for skewed features
✅ Save scaling metadata

Output:
✅ Processed CSV files
✅ Metadata JSON files
✅ Detailed logging
✅ Comprehensive documentation

Testing:
✅ Sample data generator
✅ Example usage scripts
✅ Error handling & validation

❌ FUTURE (Phase 2+):

Phase 2: Clustering
  - K-means user segmentation
  - DBSCAN for anomaly detection
  
Phase 3: Link Prediction
  - Build flight-user similarity graph
  - KNN-based link prediction
  
Phase 4: Delay Prediction
  - CNN model for weather-based delay prediction
  - Time series analysis of delays
  
Phase 5: Scoring Engine
  - Combine all signals into recommendation score
  - Ranking algorithm
  
Phase 6: Flask API
  - REST endpoints
  - Real-time recommendations
  - Caching & optimization
"""

# ============================================================================
# USAGE QUICKSTART
# ============================================================================

"""
3-STEP QUICKSTART:

1. Generate Sample Data
   $ python generate_sample_data.py
   
2. Run Preprocessing
   $ cd src
   $ python preprocessing.py
   
3. Inspect Results
   $ cd ..  # Back to root
   $ python examples.py

Expected Output:
- data/processed/users_processed.csv (with normalized features)
- data/processed/flights_processed.csv (with normalized features)
- data/processed/encoding_metadata_*.json (encoder info)
- data/processed/scaling_metadata_*.json (scaler info)
- logs/preprocessing.log (detailed execution trace)


USING THE MODULE IN YOUR CODE:

from src.preprocessing import preprocess_pipeline

report = preprocess_pipeline(
    users_path="path/to/users.csv",
    flights_path="path/to/flights.csv",
    output_dir="path/to/output"
)

print(report['processed_shapes'])  # Check output sizes
print(report['features'])  # See all created features


PRODUCTION DEPLOYMENT:

1. CI/CD Pipeline
   - Run preprocessing on schedule
   - Validate output schema
   - Alert if errors occur

2. Monitoring
   - Track data quality metrics
   - Monitor for outliers/anomalies
   - Log all transformations

3. Versioning
   - Version processed datasets
   - Version metadata files
   - Track what preprocessing version created dataset

4. Scaling
   - For larger data: use Apache Spark
   - Parallelize across multiple machines
   - Stream processing for real-time data
"""

# More documentation and ASCII diagrams available in README.md
# See also: config.py for detailed constant documentation
