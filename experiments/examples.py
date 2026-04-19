"""
Example usage of the Flight Recommendation System preprocessing module (Minimal Version).

This script demonstrates:
1. Loading raw data
2. Cleaning data
3. Encoding categorical features
4. Normalizing numerical features
5. Building interaction matrix
6. Saving processed datasets

Run this after generating sample data:
    python generate_sample_data.py
    python examples.py
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend", "preprocessing"))

from preprocess import (
    load_data,
    clean_data,
    encode_features,
    normalize_features,
    build_interaction_matrix,
    preprocess_pipeline,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
)


# ============================================================================
# Example 1: Step-by-Step Preprocessing
# ============================================================================


def example_step_by_step():
    """Demonstrate each preprocessing step individually."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: STEP-BY-STEP PREPROCESSING")
    print("=" * 80 + "\n")

    try:
        # Load data
        print("[1] Loading data...")
        users_df, flights_df, bookings_df = load_data()
        print(f"   Users shape: {users_df.shape}")
        print(f"   Flights shape: {flights_df.shape}")
        if bookings_df is not None:
            print(f"   Bookings shape: {bookings_df.shape}\n")
        else:
            print(f"   Bookings: None\n")

        # Clean data
        print("[2] Cleaning data...")
        users_df, flights_df, bookings_df = clean_data(users_df, flights_df, bookings_df)
        print(f"   Users after clean: {users_df.shape}")
        print(f"   Flights after clean: {flights_df.shape}\n")

        # Encode categorical
        print("[3] Encoding categorical features...")
        users_df, flights_df = encode_features(users_df, flights_df)
        print(f"   Users after encoding: {users_df.shape}")
        print(f"   Flights after encoding: {flights_df.shape}\n")

        # Normalize
        print("[4] Normalizing numerical features...")
        users_df, flights_df = normalize_features(users_df, flights_df)
        print(f"   Users after normalization: {users_df.shape}")
        print(f"   All numerical features scaled to [0, 1]\n")

        # Build interactions
        print("[5] Building interaction matrix...")
        interactions_df = build_interaction_matrix(bookings_df, users_df, flights_df)
        print(f"   Interactions shape: {interactions_df.shape}\n")

        print("✅ Step-by-step preprocessing complete!\n")
        return users_df, flights_df, interactions_df

    except Exception as e:
        print(f"❌ Error: {e}\n")
        return None, None, None


# ============================================================================
# Example 2: Full Pipeline (Recommended)
# ============================================================================


def example_full_pipeline():
    """Run complete preprocessing pipeline in one call."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: FULL PREPROCESSING PIPELINE (RECOMMENDED)")
    print("=" * 80 + "\n")

    try:
        report = preprocess_pipeline()

        print(f"✅ Pipeline Status: {report['status'].upper()}")
        print(f"\nDataset Shapes:")
        print(f"  Users:   {report['raw_shapes']['users']} → {report['processed_shapes']['users']}")
        print(
            f"  Flights: {report['raw_shapes']['flights']} → {report['processed_shapes']['flights']}"
        )
        print(f"  Interactions: {report['processed_shapes']['interactions']}")
        print(f"\nOutput directory: {report['output_dir']}")

        return report

    except Exception as e:
        print(f"❌ Error: {e}\n")
        return None


# ============================================================================
# Example 3: Load & Inspect Processed Data
# ============================================================================


def example_load_processed():
    """Load and inspect processed datasets."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: LOADING PROCESSED DATA")
    print("=" * 80 + "\n")

    try:
        import pandas as pd

        users_processed = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, "users_processed.csv")
        )
        flights_processed = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, "flights_processed.csv")
        )
        interactions = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, "interactions.csv")
        )

        print("Users Processed Data:")
        print(f"  Shape: {users_processed.shape}")
        print(f"  Columns: {list(users_processed.columns[:5])}... ({len(users_processed.columns)} total)")
        print(f"  Sample:\n{users_processed.iloc[0]}\n")

        print("Flights Processed Data:")
        print(f"  Shape: {flights_processed.shape}")
        print(f"  Columns: {list(flights_processed.columns)}\n")

        print("Interaction Matrix:")
        print(f"  Shape: {interactions.shape}")
        print(f"  Sample interactions:\n{interactions.head(3)}\n")

        # Verify normalization
        print("Normalization Check (sample features):")
        numerical_cols = users_processed.select_dtypes(include=["float64"]).columns
        for col in list(numerical_cols)[:3]:
            min_val = users_processed[col].min()
            max_val = users_processed[col].max()
            print(f"  {col}: [{min_val:.3f}, {max_val:.3f}]")

        print(f"\n✅ Successfully loaded processed datasets from {DATA_PROCESSED_DIR}\n")

    except FileNotFoundError:
        print(f"❌ Processed files not found in {DATA_PROCESSED_DIR}")
        print("   Run: python generate_sample_data.py")
        print("   Then: cd src && python preprocess.py\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("FLIGHT RECOMMENDATION SYSTEM - MINIMAL PREPROCESSING EXAMPLES")
    print("=" * 80)

    # Check if raw data exists
    if not os.path.exists(DATA_RAW_DIR) or not os.listdir(DATA_RAW_DIR):
        print(f"\n⚠️  Raw data not found in {DATA_RAW_DIR}")
        print("\nTo generate sample data:")
        print("  python generate_sample_data.py")
        print("\nThen run this script again.")
        return

    # Run examples
    print("\n" + "=" * 80)
    print("Running examples...")
    print("=" * 80)

    # Example 1: Step-by-step
    example_step_by_step()

    # Example 2: Full pipeline (recommended)
    example_full_pipeline()

    # Example 3: Load processed data
    example_load_processed()

    print("=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Inspect the processed CSV files in data/processed/")
    print("  2. Check the log file: logs/preprocess.log")
    print("  3. interactions.csv is ready for graph-based recommendations")
    print("  4. Ready for Phase 2: Clustering & User Segmentation")
    print()


if __name__ == "__main__":
    main()
