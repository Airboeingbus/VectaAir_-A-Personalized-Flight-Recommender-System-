# 🛫 VectaAir: Personalized Flight Recommender System

> **An intelligent, graph-based flight recommendation engine with preference learning, similarity matching, and reliability scoring.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask 2.3+](https://img.shields.io/badge/flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![Production Ready](https://img.shields.io/badge/status-production--ready-brightgreen.svg)](https://github.com/Airboeingbus/VectaAir)

---

## 🚀 Overview

VectaAir is a **full-stack flight recommendation system** that combines:
- 🔗 **Graph-based similarity matching** for personalized recommendations
- 📊 **ML-driven preference learning** from user booking history
- ⭐ **Reliability scoring** based on historical delay data
- 🎯 **Advanced preference blending** with weighted user priorities
- 🔐 **Production-grade security** with environment-based configuration
- ⚡ **Rate limiting & monitoring** for production stability

**Live Demo:** [VectaAir](https://vectaair.onrender.com) *(when deployed)*

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Smart Recommendations** | Finds flights that match user preferences (price, duration, reliability) |
| **User Clustering** | Groups similar users for collaborative filtering |
| **Graph Analysis** | Builds user-flight similarity graphs for accurate matches |
| **Preference Learning** | Learns user preferences from booking history |
| **Reliability Scoring** | Predicts flight reliability based on historical data |
| **Web Dashboard** | Clean, intuitive UI for browsing and booking flights |
| **REST API** | JSON-based API for programmatic access |
| **Secure Authentication** | Session-based login with encrypted credentials |

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Data Requirements](#data-requirements)
- [Architecture](#architecture)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Airboeingbus/VectaAir.git
cd VectaAir
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
cp .env.example .env
# Edit .env and set your FLASK_SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"
```

### Step 5: Prepare Data
```bash
# Generate sample data (optional - for testing)
python scripts/generate_sample_data.py

# Or run preprocessing on your own data
python backend/preprocessing/preprocess.py
```

---

## 🚀 Quick Start

### Development Mode
```bash
# Start Flask development server
python app.py
```
Access at: `http://localhost:5000`

### Production Mode
```bash
# Start with Gunicorn (4 workers)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### First-Time Setup Checklist
- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Configure `.env` file with secret key
- [ ] Run preprocessing on data
- [ ] Start application
- [ ] Navigate to `http://localhost:5000` and login

**Test User Credentials:**
```
user_id: U00001
password: pass123
```

---

## 📡 API Documentation

### Authentication
All endpoints except `/login` require authentication.

#### Login
```bash
POST /login
Content-Type: application/json

{
  "user_id": "U00001",
  "password": "pass123"
}

Response: 200 OK (redirects to dashboard)
```

### Recommendations

#### Basic Recommendations
```bash
POST /recommend
Authorization: Session required

{
  "user_id": "U00001"
}

Response:
{
  "status": "success",
  "recommendations": [
    {
      "flight_id": "F001",
      "price": 250.50,
      "duration": 240,
      "reliability_score": 0.95,
      "match_score": 0.87
    }
  ]
}
```

#### Advanced Recommendations with Preferences
```bash
POST /recommend_with_preferences
Authorization: Session required

{
  "origin": "BOM",
  "destination": "DEL",
  "price_pref": 70,        # 0-100 (higher = cheaper preferred)
  "time_pref": 50,         # 0-100 (higher = shorter duration)
  "reliability_pref": 80,  # 0-100 (higher = more reliable)
  "departure_time": "morning",
  "top_n": 5,
  "trip_type": "one-way",
  "departure_date": "2026-04-25"
}

Response:
{
  "status": "success",
  "recommendations": [
    {
      "flight_id": "F001",
      "price": 245.00,
      "duration": 180,
      "reliability_score": 0.96,
      "match_score": 0.92,
      "explanation": {
        "price_match": "Matches your budget preference",
        "reliability_match": "99th percentile reliability"
      }
    }
  ]
}
```

#### Health Check
```bash
GET /health

Response:
{
  "status": "ok",
  "recommender_initialized": true,
  "data_loaded": true
}
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)
```env
# Flask Configuration
FLASK_DEBUG=False                    # Disable debug in production
FLASK_SECRET_KEY=your-secret-key     # Generate with: python -c "import secrets; print(secrets.token_hex(32))"

# Server Configuration
PORT=5000
FLASK_ENV=production

# Data Configuration
DATA_PATH=data/processed

# Logging
LOG_LEVEL=INFO
```

### To Generate a Secure Secret Key
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 📊 Data Requirements

### Input Data Format

**users.csv** (in `data/raw/`)
```csv
user_id,age,gender,occupation,travel_purpose
U00001,32,Male,Engineer,Business
U00002,28,Female,Doctor,Leisure
```

**flights.csv** (in `data/raw/`)
```csv
flight_id,origin,destination,price,flight_duration_minutes,num_layovers,historical_delay_rate,airline
F001,BOM,DEL,250,240,1,0.05,AirIndia
F002,BOM,DEL,180,180,0,0.03,Indigo
```

**bookings.csv** (in `data/raw/`)
```csv
user_id,flight_id,satisfaction_rating
U00001,F001,4.5
U00002,F002,4.8
```

### Output Data (Auto-generated)
Located in `data/processed/`:
- `users_processed.csv` — Normalized user features
- `flights_processed.csv` — Normalized flight features
- `interactions.csv` — User-flight interaction matrix
- `user_clusters.csv` — Clustered users for similarity matching

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────┐
│         Flask Web App (User Interface)           │
│  ├─ Login/Authentication                         │
│  ├─ Trip Planning Dashboard                      │
│  └─ Flight Recommendations                       │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│      Flask REST API (Backend)                    │
│  ├─ /login, /logout                              │
│  ├─ /recommend                                   │
│  ├─ /recommend_with_preferences                  │
│  └─ /health                                      │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│   Recommendation Engine (Graph-based)            │
│  ├─ UserSimilarityGraph                          │
│  ├─ Preference Blending                          │
│  └─ Reliability Scoring                          │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│      Data Pipeline (ML Preprocessing)            │
│  ├─ Data Cleaning & Validation                   │
│  ├─ Feature Engineering                          │
│  ├─ Normalization & Encoding                     │
│  └─ Interaction Matrix Building                  │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│         Database Layer                           │
│  ├─ data/raw/ (input CSV files)                 │
│  └─ data/processed/ (ML-ready outputs)          │
└─────────────────────────────────────────────────┘
```

---

## 🔐 Security Features

✅ **Environment-based Configuration** - No hardcoded secrets
✅ **CSRF Protection** - Flask session management
✅ **Rate Limiting** - 10 requests/minute on expensive endpoints
✅ **Secure Session Keys** - Cryptographically generated
✅ **Input Validation** - All user inputs sanitized
✅ **Error Handling** - No stack traces exposed to users
✅ **Logging** - Production-grade logging for monitoring

---

## 📦 Project Structure

```
Flight_Recomendor/
├── app.py                          # Flask application
├── requirements.txt                # Dependencies
├── Procfile                        # Production server config
├── .env.example                    # Environment template
├── README.md                       # This file
│
├── backend/
│   ├── graph_recommender.py        # Core recommendation engine
│   ├── clustering.py               # User clustering
│   ├── config.py                   # Configuration
│   └── preprocessing/
│       ├── preprocess.py           # Data preprocessing pipeline
│       └── preprocessing.py        # Helper functions
│
├── data/
│   ├── raw/                        # Input datasets
│   │   ├── users.csv
│   │   ├── flights.csv
│   │   └── bookings.csv
│   └── processed/                  # ML-ready data
│       ├── users_processed.csv
│       ├── flights_processed.csv
│       ├── interactions.csv
│       └── user_clusters.csv
│
├── templates/
│   └── index.html                  # Web UI
│
├── static/
│   ├── styles.css                  # Stylesheets
│   └── script.js                   # Frontend logic
│
├── scripts/
│   └── generate_sample_data.py      # Sample data generator
│
└── docs/
    ├── API_DOCUMENTATION.md
    ├── ARCHITECTURE.md
    └── FLASK_WEB_APP_GUIDE.md
```

---

## 🚢 Deployment

### Deploy to Render (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Render**
   - Go to [Render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select branch: `main`

3. **Configure Environment**
   - Set environment variables in Render dashboard:
     ```
     FLASK_DEBUG=False
     FLASK_SECRET_KEY=<your-generated-key>
     PORT=5000
     ```

4. **Deploy**
   - Render will auto-deploy when you push to main
   - Access your app at: `https://<app-name>.onrender.com`

### Deploy to Heroku
```bash
heroku login
heroku create <your-app-name>
git push heroku main
heroku config:set FLASK_SECRET_KEY=<your-key>
heroku open
```

---

## 🧪 Testing

### Run Tests
```bash
pytest tests/
```

### Test Recommendations
```python
python experiments/weighted_recommender_examples.py
python experiments/clustering_examples.py
```

---

## 📈 Performance

- **Recommendation latency:** ~200-500ms
- **Concurrent users:** Supports 100+ with 4-worker Gunicorn
- **Memory footprint:** ~200MB
- **Data processing:** ~5-10 seconds for 10K users/100K flights

### Evaluation Methodology

Evaluated on leave-one-out split across 583 users.
Dataset is sparse (2000 interactions, 1000 users, 500 flights).
Sparsity is the primary limiting factor. Evaluation methodology
in experiments/recommender_analysis.ipynb.

**To Run Evaluation:**
```bash
python scripts/evaluate_recommender.py
```

### Running Analysis

To run the deep-dive analysis notebook, open a notebook session from the experiments folder and execute recommender_analysis.ipynb:

```bash
cd experiments
jupyter notebook recommender_analysis.ipynb
```

---

## 🎯 Design Decisions

### 1. Cold Start Handling Strategy

**The Problem:** New users with no booking history cannot be recommended flights using collaborative filtering (no similar users exist yet).

**Our Approach:**

The recommender has a **two-tier cold start strategy**:

1. **Graph-Based Tier (Primary)**
   - Uses cosine similarity on user profile features (age, occupation, gender, travel_purpose)
   - Even new users are assigned k=5 nearest neighbors based on feature similarity
   - Can recommend flights booked by feature-similar users
   - Works for users with profile info but no booking history

2. **Fallback Tier (When Graph Fails)**
   - If neighbors have < 3 bookings each, apply endpoint-level fallbacks:
     - Return top-reliability flights (highest on-time performance)
     - Return most-popular flights (booked by most users)
     - Return flights by preferred airline (if available in user profile)
   
   This is implemented in `app.py` at the `/recommend` endpoint level, not in the core recommender, allowing flexible fallback strategies.

**Example Cold Start Flow:**
```
User Profile: 32yo male engineer → Similar users found ✓
  ↓
Recommend flights from their bookings ✓
  (Collaborative filtering works)

User Profile: 19yo student, no bookings → Similar users found ✓
  ↓
Attempt recommendation from similar users' bookings
  ↓
If insufficient bookings: Fallback to popular/reliable flights ✓
  (Graceful degradation)
```

### 2. Cosine Similarity Design Choice

**Why Cosine Similarity Over Euclidean Distance?**

The system uses **cosine similarity** (angle between feature vectors) for user matching instead of Euclidean distance. Here's why:

| Factor | Cosine Similarity | Euclidean Distance |
|--------|------|--------|
| **High-Dimensional Data** | ✅ Robust in high dimensions | ❌ "Curse of dimensionality" |
| **One-Hot Encoded Features** | ✅ Perfect for sparse categorical features | ❌ Suffers from magnitude differences |
| **Feature Magnitude** | ✅ Magnitude-invariant (only direction matters) | ❌ Sensitive to scale |
| **Sparse Data** | ✅ Efficient and reliable | ❌ Unreliable with sparse vectors |
| **Interpretability** | ✅ Directly = correlation (-1 to 1) | ❌ Needs normalization |

**Example:**
```python
User A: [1-hot: Male, Engineer] → sparse vector after encoding
User B: [1-hot: Male, Engineer] → same sparse vector

Cosine Similarity: 1.0 (perfect match) ✓
Euclidean Distance: 0.0 (perfect match but hard to interpret) ✓

But with mixed features:
User C: [age=30, 1-hot: Female, Teacher]
User D: [age=35, 1-hot: Female, Teacher]

Cosine: ~0.95 (similar direction in high-dim space) ✓
Euclidean: 5.0 (age difference dominates) ❌
```

**Implementation:**
```python
from sklearn.metrics.pairwise import cosine_similarity

# (n_users, n_features) after StandardScaler + one-hot encoding
similarity_matrix = cosine_similarity(feature_matrix)
# Result: symmetric matrix where sim[i,j] ∈ [-1, 1]
```

### 3. Performance Optimization: Precomputed Flight Index

**The Problem:** Recommendation scoring iterates through neighbor interactions, repeatedly filtering the interactions DataFrame for each flight.

**The Solution:** **Precomputed Flight Interaction Index**

Before recommendations:
```
flight_interactions_index = {
    'F001': [('U001', 1.0), ('U003', 0.3), ...],  # 1.0 = booking, 0.3 = view
    'F002': [('U002', 1.0), ...],
    ...
}
```

During recommendation (old way):
```python
for neighbor_id in neighbors:
    # ❌ O(n) DataFrame filter for each neighbor
    neighbor_interactions = all_interactions[all_interactions['user_id'] == neighbor_id]
```

During recommendation (optimized way):
```python
for flight_id, interactions in precomputed_index.items():
    # ✅ O(1) dictionary lookup, no filtering
    for user_id, weight in interactions:
        if user_id in neighbor_dict:
            score += similarity[user_id] * weight
```

**Impact:** ~2-3x faster recommendations with large datasets (10K+ users).

---

## 🐛 Troubleshooting

### Issue: "Recommender not initialized"
```
Solution: Check that data files exist in data/processed/
Run: python backend/preprocessing/preprocess.py
```

### Issue: "FLASK_SECRET_KEY not set"
```
Solution: 
1. Create .env file from .env.example
2. Generate key: python -c "import secrets; print(secrets.token_hex(32))"
3. Set FLASK_SECRET_KEY in .env
```

### Issue: Port already in use
```
Solution: 
Change PORT in .env or run: export PORT=5001
```

---

## 📚 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Reference](docs/API_DOCUMENTATION.md)
- [Flask Web App Guide](docs/FLASK_WEB_APP_GUIDE.md)
- [Hybrid Reliability Guide](docs/HYBRID_RELIABILITY_GUIDE.md)

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **Airboeingbus** - Initial development
- Contributors welcome!
---

## 📧 Contact & Support

- **GitHub Issues:** [Report bugs here](https://github.com/Airboeingbus/VectaAir/issues)
- **Discussions:** [Join our community](https://github.com/Airboeingbus/VectaAir/discussions)
---
