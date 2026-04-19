# 🛫 Flight Recommendation System - Minimal Preprocessing Backend (Phase 1 Refactored)

A **clean, lightweight, production-ready** preprocessing module optimized for **graph-based flight recommendations**.

- ✅ Minimal & focused (~260 lines)
- ✅ No overengineering
- ✅ Graph-ready: builds interaction matrix
- ✅ Fast & simple pipeline
- ✅ Inclusive design (one-hot gender, no stereotypes)

---

## 🎯 What It Does

Transforms raw flight booking data into ML-ready datasets for clustering and similarity-based recommendations.

**Pipeline:**
```
Raw Data → Clean → Encode Categorical → Normalize → Build Interactions → Save
```

---

## 📦 Input Data

Place CSV files in `data/raw/`:

**users.csv** (Required)
```
user_id, age, gender, occupation, travel_purpose, ...
U001, 32, Male, Engineer, Business, ...
U002, 28, Female, Doctor, Leisure, ...
```

**flights.csv** (Required)
```
flight_id, price, flight_duration_minutes, num_layovers, historical_delay_rate, ...
F001, 250, 240, 1, 0.05, ...
F002, 180, 180, 0, 0.03, ...
```

**bookings.csv** (Optional but recommended)
```
user_id, flight_id, satisfaction_rating  (or just user_id, flight_id)
U001, F001, 4.5
U002, F002, 4.8
```

---

## 🚀 Quick Start

```bash
# 1. Generate sample data
python generate_sample_data.py

# 2. Run preprocessing
cd src && python preprocess.py

# 3. Check results
ls data/processed/
```

**Output:**
- `data/processed/users_processed.csv` — ML-ready user features
- `data/processed/flights_processed.csv` — ML-ready flight features
- `data/processed/interactions.csv` — User-flight interaction matrix

---

## 📊 Features

### User Features (Minimal)
- `age` (normalized to [0, 1])
- `gender` (one-hot: Male, Female, Non-binary, Prefer not to say)
- `occupation` (one-hot)
- `travel_purpose` (one-hot)

### Flight Features (Minimal)
- `price` (normalized)
- `flight_duration_minutes` (normalized)
- `num_layovers` (count)
- `historical_delay_rate` (normalized)

### Interaction Matrix
- `user_id, flight_id, interaction`
- Where `interaction` = booking rating (if available) or 1 (booked)
- **Critical for graph construction** 🔗

---

## 🔧 API Reference

### Main Functions

```python
from preprocess import preprocess_pipeline

# One-liner: run everything
report = preprocess_pipeline()
```

#### Step-by-Step

```python
from preprocess import load_data, clean_data, encode_features, normalize_features, build_interaction_matrix

users_df, flights_df, bookings_df = load_data()
users_df, flights_df, bookings_df = clean_data(users_df, flights_df, bookings_df)
users_df, flights_df = encode_features(users_df, flights_df)
users_df, flights_df = normalize_features(users_df, flights_df)
interactions_df = build_interaction_matrix(bookings_df, users_df, flights_df)
```

### Functions Explained

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `load_data()` | Load CSV files | paths | DataFrames |
| `clean_data()` | Handle missing, duplicates, validate | df | clean df |
| `encode_features()` | One-hot encode categorical | df | numeric df |
| `normalize_features()` | MinMax scale to [0,1] | df | normalized df |
| `build_interaction_matrix()` | Create user-flight graph | bookings_df | interactions |
| `preprocess_pipeline()` | Run all steps | paths | report |

---

## 📋 Output Format

### users_processed.csv
```
N_users × ~20 features
All numerical features ∈ [0, 1]
Sample row:
  user_id, age, gender_Male, gender_Female, occupation_Engineer, ...
  U001, 0.42, 1.0, 0.0, 1.0, ...
```

### flights_processed.csv
```
N_flights × 4 features
All numerical features ∈ [0, 1]
Columns: flight_id, price, flight_duration_minutes, num_layovers, historical_delay_rate
```

### interactions.csv
```
user_id, flight_id, interaction
U001, F001, 1
U002, F003, 4.5  (if ratings provided)
...
```

---

## 🎓 Design Highlights

✅ **Minimal & Clear**
- ~260 lines (no bloat)
- 6 core functions
- Easy to understand and extend

✅ **No Overengineering**
- No unnecessary derived features
- No complex config files
- Just what you need for graphs

✅ **Graph-Ready**
- Interaction matrix as standard output
- Perfect for GNN, node2vec, link prediction

✅ **Inclusive**
- Gender: one-hot (not binary)
- Occupation: feature (not stereotype)
- Budget: data-driven (if added later)

✅ **Production Code**
- Error handling
- Logging
- Type hints
- Clean structure

---

## 🔍 Data Transformations

**Age:**
```
Raw: 32 (from user)
→ Normalized: 0.421 (in [0, 1])
```

**Gender:**
```
Raw: "Male"
→ Encoded: gender_Male=1.0, gender_Female=0.0, gender_Non-binary=0.0, gender_Other=0.0
```

**Price:**
```
Raw: 250 (USD)
→ Normalized: 0.18 (in [0, 1])
```

**Interactions:**
```
Raw (bookings): U001 booked F001 with satisfaction 4.5
→ interactions.csv: U001, F001, 4.5
→ Use for training graph embeddings!
```

---

## 📁 Project Structure

```
Flight_Recomendor/
├── src/
│   ├── preprocess.py           # Main preprocessing module (NEW)
│   └── __init__.py
├── data/
│   ├── raw/                    # Place input CSVs here
│   └── processed/              # Outputs go here
├── logs/
│   └── preprocess.log          # Execution log
├── examples.py                 # Usage examples
├── generate_sample_data.py     # Synthetic data generator
└── README.md                   # This file
```

---

## 🧪 Testing

```bash
# Generate sample data
python generate_sample_data.py

# Run preprocessing
cd src && python preprocess.py

# Check output
cd .. && python examples.py

# Inspect files
head -5 data/processed/users_processed.csv
head -5 data/processed/interactions.csv
```

---

## 📊 Example Report

After running the pipeline:

```
PREPROCESSING COMPLETE
Users: (1000, 9) → (1000, 20)
Flights: (500, 8) → (500, 4)
Interactions: (2000, 3)
```

- 1000 users × 20 features (age + one-hot encoded)
- 500 flights × 4 features (all normalized)
- 2000 user-flight interactions (ready for graphs!)

---

## 🔗 Next Steps: Phase 2

With this minimal preprocessing, you're ready for:

1. **User Clustering** (K-means on user features)
2. **Similarity Graphs** (KNN on user features)
3. **Link Prediction** (predict missing edges in user-flight graph)
4. **Graph Embeddings** (node2vec, GCN)
5. **Recommendations** (similarity-based ranking)

```python
# Example (later): Use interaction matrix to build graph
import networkx as nx

G = nx.Graph()
G.add_weighted_edges_from(interactions.values)  # User-flight bipartite graph
```

---

## 💡 Why Minimal?

❌ **Old approach (too complex):**
- Derived features: budget_sensitivity, loyalty_score, seasonality
- Feature metadata saved as JSON
- Complex config system
- Not needed for graph-based recommendations

✅ **New approach (minimal):**
- Only features needed for clustering & similarity
- Interaction matrix as graph edges
- Simple, clean code
- Fast, understandable, extensible

---

## ⚙️ Configuration (In code)

Edit `src/preprocess.py` to customize:

```python
# Feature lists
USER_FEATURES = ["age", "gender", "occupation", "travel_purpose"]
FLIGHT_FEATURES = ["price", "flight_duration_minutes", "num_layovers", "historical_delay_rate"]

# Encoding
ONE_HOT_USER_FEATURES = ["gender", "occupation", "travel_purpose"]

# Normalization
NORMALIZE_FEATURES = ["age", "price", "flight_duration_minutes", "historical_delay_rate"]

# Validation
AGE_RANGE = (18, 100)
MIN_PRICE = 0
```

No separate config file—everything inline for clarity!

---

## 📝 Logging

All operations logged to `logs/preprocess.log`:

```
2026-04-18 14:32:15 - INFO - Loading data from data/raw
2026-04-18 14:32:15 - INFO - Loaded users: 1000 rows
2026-04-18 14:32:15 - INFO - Loaded flights: 500 rows
...
2026-04-18 14:32:18 - INFO - PREPROCESSING COMPLETE
```

---

## ✅ Checklist

- [x] Load raw data (users, flights, bookings)
- [x] Clean: missing values, duplicates, validation
- [x] Encode: categorical → one-hot
- [x] Normalize: numerical → [0, 1]
- [x] Build: interaction matrix (user-flight graph)
- [x] Save: processed datasets
- [x] Minimal code (~260 lines)
- [x] No overengineering
- [x] Graph-ready

---

## 🎯 Status

✅ **Phase 1 (Refactored) - COMPLETE**  
Ready for: Phase 2 - Clustering & Similarity Graphs

---

**Total Lines:** ~260 (production code)  
**Dependencies:** pandas, numpy, scikit-learn  
**Status:** Ready for graph-based recommendations 🔗

