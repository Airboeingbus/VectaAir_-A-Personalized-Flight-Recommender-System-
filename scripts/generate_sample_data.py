"""
Generate sample datasets for testing the Flight Recommendation System.

This script creates realistic synthetic data for users and flights to enable
testing the preprocessing pipeline without real production data.

Usage:
    python generate_sample_data.py
"""

import os
import csv
import random
from datetime import datetime
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

NUM_USERS = 1000
NUM_FLIGHTS = 500

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(DATA_RAW_DIR, exist_ok=True)

# ============================================================================
# Sample Data
# ============================================================================

OCCUPATIONS = [
    "Engineer",
    "Accountant",
    "Manager",
    "Doctor",
    "Teacher",
    "Software Developer",
    "Data Scientist",
    "Consultant",
    "Sales Executive",
    "Entrepreneur",
    "Student",
    "Retired",
]

EDUCATION_LEVELS = ["High School", "Bachelors", "Masters", "PhD"]

TRAVEL_PURPOSES = ["Business", "Leisure", "Family Visit", "Conference", "Holiday"]

GENDERS = ["Male", "Female", "Non-binary", "Prefer not to say"]

AIRLINES = ["United", "Delta", "American", "Southwest", "Alaska", "JetBlue", "Spirit"]

SEAT_CLASSES = ["Economy", "Premium Economy", "Business", "First"]

DESTINATIONS = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "London",
    "Tokyo",
    "Dubai",
    "Singapore",
    "Sydney",
]


# ============================================================================
# Generator Functions
# ============================================================================


def generate_users(num_users: int = NUM_USERS) -> list:
    """Generate synthetic user profiles."""
    users = []
    for i in range(num_users):
        user = {
            "user_id": f"U{i+1:05d}",
            "age": random.randint(18, 75),
            "gender": random.choice(GENDERS),
            "occupation": random.choice(OCCUPATIONS),
            "education": random.choice(EDUCATION_LEVELS),
            "travel_purpose": random.choice(TRAVEL_PURPOSES),
            "num_bookings": random.randint(0, 50),
            "preferred_airline": random.choice(AIRLINES),
            "loyalty_tier": random.choice(["Bronze", "Silver", "Gold", "Platinum"]),
        }
        users.append(user)
    return users


def generate_flights(num_flights: int = NUM_FLIGHTS) -> list:
    """Generate synthetic flight listings."""
    flights = []
    for i in range(num_flights):
        flight = {
            "flight_id": f"F{i+1:05d}",
            "airline": random.choice(AIRLINES),
            "origin": "New York",  # Simplified
            "destination": random.choice(DESTINATIONS),
            "price": round(np.random.lognormal(5, 0.8), 2),  # Log-normal price distribution
            "booking_lead_time": random.randint(1, 90),  # Days before departure
            "flight_duration_minutes": random.randint(60, 600),
            "num_layovers": random.randint(0, 3),
            "airline_rating": round(random.uniform(3.0, 5.0), 1),
            "seat_class": random.choice(SEAT_CLASSES),
            "historical_delay_rate": round(random.uniform(0.0, 0.2), 3),
            "route_popularity": random.randint(10, 200),  # Flights per month on this route
            "departure_month": random.randint(1, 12),
        }
        flights.append(flight)
    return flights


def generate_bookings(users: list, flights: list, num_bookings: int = 2000) -> list:
    """Generate synthetic booking history (user-flight interactions)."""
    bookings = []
    for i in range(min(num_bookings, len(users) * 5)):  # Limit bookings
        booking = {
            "booking_id": f"B{i+1:06d}",
            "user_id": random.choice(users)["user_id"],
            "flight_id": random.choice(flights)["flight_id"],
            "booking_date": f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            "satisfaction_rating": random.choice([1, 2, 3, 4, 5]),
            "actual_delay_minutes": random.randint(-15, 120),  # Negative = early departure
        }
        bookings.append(booking)
    return bookings


# ============================================================================
# Save to CSV
# ============================================================================


def save_to_csv(data: list, filename: str) -> None:
    """Save list of dicts to CSV file."""
    if not data:
        print(f"⚠️  {filename}: No data to save")
        return

    filepath = os.path.join(DATA_RAW_DIR, filename)
    fieldnames = data[0].keys()

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ {filename}: {len(data)} rows saved to {filepath}")


# ============================================================================
# Main
# ============================================================================


def main():
    """Generate and save all sample datasets."""
    print("=" * 80)
    print("GENERATING SAMPLE DATA FOR FLIGHT RECOMMENDATION SYSTEM")
    print("=" * 80)
    print()

    # Generate
    print(f"Generating {NUM_USERS} users...")
    users = generate_users(NUM_USERS)

    print(f"Generating {NUM_FLIGHTS} flights...")
    flights = generate_flights(NUM_FLIGHTS)

    print(f"Generating booking history...")
    bookings = generate_bookings(users, flights, num_bookings=2000)

    print()

    # Save
    print("Saving to CSV...")
    save_to_csv(users, "users.csv")
    save_to_csv(flights, "flights.csv")
    save_to_csv(bookings, "bookings.csv")

    print()
    print("=" * 80)
    print("✅ SAMPLE DATA GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"Data location: {DATA_RAW_DIR}")
    print()
    print("Next steps:")
    print("  1. cd src")
    print("  2. python preprocessing.py")
    print()


if __name__ == "__main__":
    main()
