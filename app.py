"""
Flight Recommendation System - Flask Backend API

Simple Flask API for the hybrid flight recommender system.
Exposes endpoints to get recommendations and health status.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables from .env file
load_dotenv()

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠ Warning: pandas not installed. Some features may be limited.")

from flask import Flask, jsonify, request, render_template, session, redirect, url_for
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from graph_recommender import UserSimilarityGraph

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Session secret - from environment variable (required for security)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    if os.environ.get('FLASK_DEBUG', 'False') != 'True':
        # In production, this is a critical error
        raise ValueError(
            "FLASK_SECRET_KEY environment variable not set! "
            "Please set FLASK_SECRET_KEY in your .env file or as an environment variable."
        )
    else:
        # Development fallback - warn but don't fail
        app.secret_key = 'dev-key-change-in-production'
        logger.warning("Using development secret key. Set FLASK_SECRET_KEY for production.")

# Rate limiting to prevent DOS attacks on expensive endpoints
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Global recommender instance
recommender = None

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def initialize_recommender():
    """Initialize the recommender system on startup."""
    global recommender
    try:
        logger.info("Initializing recommender system...")
        recommender = UserSimilarityGraph(k_neighbors=5)
        recommender.load_data()
        recommender.build_feature_matrix()
        recommender.compute_similarity()
        recommender.build_graph()
        logger.info("✓ Recommender initialized successfully")
        return True
    except FileNotFoundError as e:
        logger.error(f"✗ Data files not found during recommender initialization: {e}")
        logger.error("Ensure all required data files exist in data/processed/")
        return False
    except Exception as e:
        logger.error(f"✗ Error initializing recommender: {e}", exc_info=True)
        return False


@app.route('/')
def home():
    """Always redirect to login page."""
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handle login page and authentication.
    
    GET: Serve the login page
    POST: Handle login credentials
    
    POST Request body:
    {"user_id": "U00001", "password": "pass123"}
    
    POST Response:
    {"success": true} or {"success": false, "error": "Invalid credentials"}
    """
    if request.method == 'GET':
        return render_template('login.html')
    
    # POST method
    try:
        data = request.get_json()
        user_id = str(data.get('user_id', '')).strip()
        password = str(data.get('password', '')).strip()
        
        logger.debug(f"Login attempt: user_id='{user_id}'")
        logger.debug(f"Available users: {list(users_db.keys())}")
        
        if not user_id or not password:
            return jsonify({"success": False, "error": "Missing credentials"}), 400
        
        if user_id not in users_db:
            logger.warning(f"User {user_id} not found in database")
            return jsonify({"success": False, "error": "Invalid credentials"}), 401
        
        stored_data = users_db[user_id]
        stored_password = str(stored_data['password']).strip() if isinstance(stored_data, dict) else str(stored_data).strip()
        
        logger.debug(f"Stored password check for user {user_id}")
        
        if stored_password == password:
            session['user_id'] = user_id
            logger.info(f"Login successful for {user_id}")
            return jsonify({"success": True, "user_id": user_id, "redirect": url_for('search')}), 200
        else:
            logger.warning(f"Password mismatch for {user_id}")
            return jsonify({"success": False, "error": "Invalid credentials"}), 401
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/search', methods=['GET'])
@login_required
def search():
    """Serve the search page (requires login)."""
    user_id = session.get('user_id')
    user_pref = users_db.get(user_id, {})
    reliability = user_pref.get('reliability', 'medium') if isinstance(user_pref, dict) else 'medium'
    return render_template('index.html', user_id=user_id, reliability_pref=reliability)


@app.route('/logout', methods=['GET'])
def logout():
    """Logout the user."""
    session.clear()
    return redirect(url_for('login'))


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route('/recommend', methods=['POST'])
@limiter.limit("10 per minute")
def recommend():
    """
    Get flight recommendations for a user.
    
    Request body:
    {
        "user_id": "U00001"
    }
    
    Response:
    {
        "user_id": "U00001",
        "fallback": false,
        "recommendations": [
            {"flight_id": "F001", "score": 0.85, "reliability": 0.95},
            {"flight_id": "F023", "score": 0.78, "reliability": 0.92}
        ]
    }
    """
    if not recommender:
        return jsonify({"error": "Recommender not initialized"}), 500
    
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        top_n = data.get('top_n', 5)
        
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        
        # Check if user exists in graph
        neighbors = recommender.get_neighbors(user_id)
        if not neighbors:
            return jsonify({"error": "User not found"}), 404
        
        # Get recommendations
        recommendations = recommender.recommend(user_id, top_n=top_n)
        
        # Track if using fallback
        is_fallback = False
        
        # If no recommendations, return most popular flights as fallback
        if not recommendations:
            recommendations = _get_popular_fallback(top_n)
            is_fallback = True
        
        # Format response with reliability
        result = {
            "user_id": user_id,
            "fallback": is_fallback,
            "recommendations": [
                {
                    "flight_id": flight_id,
                    "score": round(float(score), 3),
                    "reliability": round(recommender.reliability_scores.get(flight_id, 1.0), 3)
                }
                for flight_id, score in recommendations
            ]
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _get_popular_fallback(top_n: int = 5):
    """
    Fallback: Return most popular flights when no personalized recommendations available.
    
    Args:
        top_n: Number of popular flights to return
    
    Returns:
        List of (flight_id, score) tuples based on booking frequency
    """
    if not recommender or recommender.interactions is None:
        return []
    
    # Count bookings per flight
    flight_popularity = (
        recommender.interactions[recommender.interactions['interaction_type'] == 'booking']
        .groupby('flight_id')
        .size()
        .sort_values(ascending=False)
        .head(top_n)
    )
    
    # Return as (flight_id, popularity_score) tuples
    return [(flight_id, float(count)) for flight_id, count in flight_popularity.items()]


# Load users CSV
def load_users():
    """Load user credentials and preferences from CSV."""
    try:
        # Try with pandas first
        import pandas as pd
        users_df = pd.read_csv('data/users.csv')
        users_dict = {}
        for _, row in users_df.iterrows():
            user_id = str(row['user_id']).strip()
            password = str(row['password']).strip()
            reliability = str(row.get('reliability', 'medium')).strip() if 'reliability' in row else 'medium'
            users_dict[user_id] = {
                'password': password,
                'reliability': reliability
            }
        print(f"✓ Loaded {len(users_dict)} users from CSV using pandas")
        return users_dict
    except ImportError:
        print("⚠ Pandas not available, loading users.csv without pandas...")
        # Fallback: load CSV without pandas
        try:
            users_dict = {}
            with open('data/users.csv', 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    header = lines[0].strip().split(',')
                    for line in lines[1:]:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            user_id = parts[0].strip()
                            password = parts[1].strip()
                            users_dict[user_id] = {
                                'password': password,
                                'reliability': 'medium'
                            }
            print(f"✓ Loaded {len(users_dict)} users from CSV")
            return users_dict
        except Exception as e:
            print(f"✗ Error loading users: {e}")
            return {}
    except Exception as e:
        print(f"✗ Error loading users: {e}")
        return {}


# Load airports CSV
def load_airports():
    """Load airports from CSV."""
    if HAS_PANDAS:
        try:
            return pd.read_csv('data/processed/airports_cleaned.csv')
        except Exception as e:
            print(f"Error loading airports with pandas: {e}")
            return pd.DataFrame()
    else:
        # Fallback: load as list of dicts without pandas
        try:
            airports = []
            with open('data/processed/airports_cleaned.csv', 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    header = lines[0].strip().split(',')
                    for line in lines[1:]:
                        # Handle CSV with possible quoted fields
                        parts = [p.strip('"').strip() for p in line.strip().split(',')]
                        if len(parts) >= 5:
                            airport_dict = {
                                'airport_name': parts[0],
                                'city': parts[1],
                                'iata_code': parts[2],
                                'iso_country': parts[3],
                                'display': parts[4]
                            }
                            airports.append(airport_dict)
            print(f"Loaded {len(airports)} airports without pandas")
            return airports
        except Exception as e:
            print(f"Error loading airports: {e}")
            return []


users_db = load_users()
airports_df = load_airports()


@app.route('/search_airports', methods=['GET'])
def search_airports():
    """
    Search airports by city or IATA code.
    
    Query param: q (search term)
    Response: [{"display": "Mumbai (BOM)", "iata_code": "BOM"}, ...]
    """
    try:
        query = request.args.get('q', '').upper()
        
        if not query or len(query) < 2:
            return jsonify([]), 200
        
        if HAS_PANDAS and hasattr(airports_df, 'columns'):
            # Pandas DataFrame approach
            mask = (airports_df['city'].str.upper().str.contains(query, na=False)) | \
                   (airports_df['iata_code'].str.contains(query, na=False))
            results = airports_df[mask][['display', 'iata_code']].head(5).to_dict('records')
        else:
            # Fallback: search in list of dicts
            results = []
            for airport in airports_df:
                if (query in airport.get('city', '').upper() or 
                    query in airport.get('iata_code', '').upper()):
                    results.append({
                        'display': airport.get('display', ''),
                        'iata_code': airport.get('iata_code', '')
                    })
                if len(results) >= 5:
                    break
        
        return jsonify(results), 200
    
    except Exception as e:
        print(f"Search airports error: {e}")
        return jsonify({"error": str(e)}), 500


def compute_preference_matching(flight_id, user_price_pref, user_time_pref, flights_df):
    """
    Compute how well a flight matches user's stated preferences.
    
    Maps user preferences (0-100) against actual flight attributes (normalized 0-1).
    
    Args:
        flight_id: Flight identifier
        user_price_pref: User's price preference (0-100)
        user_time_pref: User's time preference (0-100)
        flights_df: DataFrame of flight data
    
    Returns:
        matching_score: 0-1, higher = better match
    """
    try:
        flight = flights_df[flights_df['flight_id'] == flight_id]
        if flight.empty:
            return 0.5  # Default if flight not found
        
        flight = flight.iloc[0]
        
        # Normalize user preferences to 0-1
        user_price_norm = user_price_pref / 100.0
        user_time_norm = user_time_pref / 100.0
        
        # Get flight attributes (already normalized in processed data)
        flight_price = float(flight.get('price', 0.5))  # 0-1 normalized
        flight_duration = float(flight.get('flight_duration_minutes', 0.5))  # 0-1 normalized
        
        # Compute affinity (inverse distance)
        # price: 0 = cheap pref, 1 = premium pref. distance = abs(user_pref - flight_price)
        price_affinity = 1.0 - abs(user_price_norm - flight_price)
        
        # time: 0 = fast pref, 1 = comfort pref. distance = abs(user_pref - flight_duration_norm)
        time_affinity = 1.0 - abs(user_time_norm - flight_duration)
        
        # Weighted average (60% price, 40% time)
        matching_score = 0.6 * price_affinity + 0.4 * time_affinity
        matching_score = max(0.0, min(1.0, matching_score))  # Clamp to [0, 1]
        
        return matching_score
    
    except Exception as e:
        print(f"Error computing preference matching for {flight_id}: {e}")
        return 0.5  # Default fallback


@app.route('/recommend_with_preferences', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def recommend_with_preferences():
    """
    Get recommendations with advanced preference weighting and explanations.
    
    Request body:
    {
        "origin": "BOM",
        "destination": "DEL",
        "price_pref": 50,
        "time_pref": 50,
        "reliability_pref": 50,
        "departure_time": "any",
        "top_n": 5,
        "trip_type": "one-way",
        "departure_date": "2026-04-20",
        "return_date": "2026-04-22"
    }
    """
    if not recommender:
        return jsonify({"error": "Recommender not initialized"}), 500
    
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        origin = data.get('origin', '').strip().upper()
        destination = data.get('destination', '').strip().upper()
        price_pref = int(data.get('price_pref', 50))
        time_pref = int(data.get('time_pref', 50))
        reliability_pref = int(data.get('reliability_pref', 50))
        departure_time = data.get('departure_time', 'any').lower()
        top_n = int(data.get('top_n', 5))
        trip_type = data.get('trip_type', 'one-way')
        departure_date = data.get('departure_date')
        return_date = data.get('return_date')
        
        if not origin or not destination:
            return jsonify({"error": "Missing origin or destination"}), 400
        
        if origin == destination:
            return jsonify({"error": "Origin and destination must be different"}), 400
        
        # Check if user exists in recommender
        neighbors = recommender.get_neighbors(user_id)
        if not neighbors:
            return jsonify({"error": "User not found"}), 404
        
        # Prepare user preferences dict for preference-boosted recommendations
        user_preferences = {
            'price_pref': price_pref,
            'time_pref': time_pref,
            'reliability_pref': reliability_pref
        }
        
        # Validate coherence: blend stated preferences with behavioral preferences
        # Users whose stated prefs don't match behavior will have their behavior weighted more
        blended_preferences = recommender.blend_preferences(user_id, user_preferences)
        user_preferences.update(blended_preferences)  # Use blended for all subsequent logic
        
        # Get base recommendations with preference-based neighbor reweighting
        recommendations = recommender.recommend_with_preference_boost(
            user_id, user_preferences, top_n=top_n * 2
        )
        
        # If no recommendations, return most popular flights as fallback
        is_fallback = False
        if not recommendations:
            recommendations = _get_popular_fallback(top_n)
            is_fallback = True
        
        # Load flights data for preference matching (if available)
        try:
            flights_df = recommender.flights  # Use flights data from recommender
        except:
            flights_df = None
        
        # Normalize preferences (0-100 to 0-1)
        price_norm = price_pref / 100
        time_norm = time_pref / 100
        reliability_norm = reliability_pref / 100
        
        # Date-based adjustment (simple boost)
        date_factor = 0.0
        if departure_date:
            date_factor += 0.05  # Base boost for having selected dates
        if trip_type == 'round-trip' and return_date:
            date_factor += 0.03  # Additional boost for round-trip
        
        # Score breakdown and explanations
        scored_recommendations = []
        for flight_id, graph_score in recommendations:
            # Get flight reliability (0-1 normalized)
            reliability_val = recommender.reliability_scores.get(flight_id, 1.0)
            
            # Compute reliability component weighted by user preference
            # Reliability component is: reliability_value * user_preference_weight
            reliability_score = reliability_val * reliability_norm
            
            # Compute per-flight preference matching (NEW: feature-based, not static)
            # Compares user's stated preferences against actual flight attributes
            if flights_df is not None:
                preference_score = compute_preference_matching(
                    flight_id, price_pref, time_pref, flights_df
                )
            else:
                preference_score = 0.5  # Default if flights data unavailable
            
            # FIXED SCORING: 0.6*collab + 0.2*reliability + 0.1*prefs + 0.1*bonus
            final_score = (
                0.6 * graph_score +         # Collaborative filtering (pure graph score)
                0.2 * reliability_score +   # Reliability component (NOT double-counted)
                0.1 * preference_score +    # Preference alignment bonus
                0.1 * date_factor           # Trip planning bonus
            )
            final_score = min(1.0, final_score)  # Cap at 1.0
            
            # Generate explanation
            # Generate explanation from actual score breakdown
            explanation = _generate_explanation(
                graph_score,           # Collaborative filtering component (will be weighted 0.6)
                reliability_score,     # Reliability component (will be weighted 0.2)
                preference_score,      # Preference matching (will be weighted 0.1)
                date_factor            # Trip planning bonus (will be weighted 0.1)
            )
            
            scored_recommendations.append((
                flight_id, final_score, graph_score, 
                reliability_score, preference_score, date_factor, explanation
            ))
        
        # Re-sort by final score
        scored_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N
        final_recommendations = scored_recommendations[:top_n]
        
        # Format response with breakdown and explanation
        result = {
            "recommendations": [
                {
                    "flight_id": flight_id,
                    "score": round(float(final_score), 3),
                    "breakdown": {
                        "collaborative": round(float(0.6 * graph_score), 3),
                        "reliability": round(float(0.2 * reliability_score), 3),
                        "preference": round(float(0.1 * preference_score), 3),
                        "bonus": round(float(0.1 * date_factor), 3)
                    },
                    "explanation": explanation
                }
                for flight_id, final_score, graph_score, reliability_score, preference_score, date_factor, explanation in final_recommendations
            ],
            "fallback": is_fallback,
            "message": "Recommendations based on popular routes." if is_fallback else ""
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"Recommendation error: {e}")
        return jsonify({"error": str(e)}), 500


def _generate_explanation(graph_score: float, reliability_component: float, 
                         preference_score: float, date_bonus: float) -> str:
    """
    Generate human-readable explanation based on actual score breakdown.
    
    Derives explanation from the computed components that contributed most to the score.
    
    Args:
        graph_score: Collaborative filtering score (0-1, unadjusted, will be weighted 0.6)
        reliability_component: Weighted reliability score (0-1, will be weighted 0.2)
        preference_score: Feature-based preference matching (0-1, will be weighted 0.1)
        date_bonus: Trip planning bonus (0-1, will be weighted 0.1)
    
    Returns:
        explanation: Human-readable explanation tied to actual computed values
    """
    reasons = []
    
    # Compute weighted contributions to final score
    collab_contrib = 0.6 * graph_score
    reliability_contrib = 0.2 * reliability_component
    pref_contrib = 0.1 * preference_score
    bonus_contrib = 0.1 * date_bonus
    
    # Find which factors contributed most
    contributions = [
        ('collab', collab_contrib, 'Users similar to you frequently book this flight'),
        ('reliability', reliability_contrib, 'This flight has strong on-time performance'),
        ('preference', pref_contrib, 'This flight matches your stated preferences'),
        ('bonus', bonus_contrib, 'Matches your trip dates and type')
    ]
    
    # Sort by contribution magnitude
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    # Add top 2-3 reasons (only those with meaningful contribution > 0.05)
    for factor_name, contrib_value, reason_text in contributions:
        if contrib_value > 0.05:  # Only include if materially contributing
            reasons.append(reason_text)
        if len(reasons) >= 3:  # Limit to top 3 reasons
            break
    
    # Fallback if no strong contributors
    if not reasons:
        return "Recommended based on your travel history and stated preferences."
    
    # Generate explanation
    if len(reasons) == 1:
        return f"Recommended because: {reasons[0]}."
    else:
        return "Recommended because: " + ", ".join(reasons) + "."


def update_user_preference(user_id: str, reliability: str):
    try:
        if HAS_PANDAS:
            users_df = pd.read_csv('data/users.csv')
            if user_id in users_df['user_id'].values:
                users_df.loc[users_df['user_id'] == user_id, 'reliability'] = reliability
                users_df.to_csv('data/users.csv', index=False)
                users_db[user_id]['reliability'] = reliability
        else:
            # Fallback: update CSV without pandas
            with open('data/users.csv', 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 0:
                header = lines[0].strip()
                updated_lines = [header + '\n']
                
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 2 and parts[0].strip() == user_id:
                        # Update this user's reliability
                        updated_lines.append(f"{parts[0]},{parts[1]},{reliability}\n")
                        users_db[user_id]['reliability'] = reliability
                    else:
                        updated_lines.append(line)
                
                with open('data/users.csv', 'w') as f:
                    f.writelines(updated_lines)
    except Exception as e:
        print(f"Error updating user preference: {e}")


if __name__ == '__main__':
    # Initialize recommender on startup
    if not initialize_recommender():
        logger.warning("Warning: Recommender initialization failed. Some features may be unavailable.")
    
    # Configuration from environment
    debug = os.environ.get('FLASK_DEBUG', 'False') == 'True'
    port = int(os.environ.get('PORT', 5000))
    
    if debug:
        logger.warning("Running in DEBUG mode. Disable this for production!")
    
    # Run Flask app
    app.run(debug=debug, host='0.0.0.0', port=port)
