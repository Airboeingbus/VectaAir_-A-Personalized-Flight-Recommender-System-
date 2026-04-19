"""
Hybrid Graph Recommender with Weighted Similarity + Flight Reliability

Builds a user similarity graph using cosine similarity, then recommends flights
using weighted scoring based on neighbor similarity, interactions, and flight reliability.

Pipeline:
1. Load user features and interaction history
2. Load flight reliability data (1 - historical_delay_rate)
3. Compute cosine similarity between all users
4. Build user-user similarity graph with similarity scores
5. For recommendations: Score(flight) = sum(similarity × interaction) × reliability
6. Rank flights by final weighted score

Output:
- User similarity graph with similarity scores
- Flight reliability scores
- Hybrid recommendations: recommend(user_id, top_n=5)
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

# Configuration
K_NEIGHBORS = 5
INTERACTION_WEIGHT_BOOKING = 1.0  # Weight for booking interaction
INTERACTION_WEIGHT_VIEW = 0.3     # Weight for view interaction
RANDOM_STATE = 42

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class UserSimilarityGraph:
    """
    Weighted graph recommender using cosine similarity scoring.
    
    For each user, recommends flights using:
    Score(f) = sum over neighbors: similarity(user, neighbor) * interaction(neighbor, f)
    """
    
    def __init__(self, k_neighbors: int = K_NEIGHBORS):
        """
        Initialize the hybrid graph recommender.
        
        Args:
            k_neighbors: Number of nearest neighbors per user
        """
        self.k_neighbors = k_neighbors
        self.users = None
        self.flights = None  # NEW: Store flights data
        self.interactions = None
        self.similarity_matrix = None
        self.user_index = {}  # user_id → index mapping
        self.neighbors_with_scores = {}  # user_id → [(neighbor_id, similarity_score)]
        self.reliability_scores = {}  # flight_id → reliability (1 - delay_rate)
        logger.info(f"Initialized HybridGraphRecommender (k={k_neighbors})")
    
    
    def load_data(self) -> None:
        """Load user features, interaction history, and flight reliability data."""
        import os
        
        # Handle both direct and src-directory execution
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        users_path = os.path.join(base_path, 'data/processed/users_processed.csv')
        interactions_path = os.path.join(base_path, 'data/processed/interactions.csv')
        flights_path = os.path.join(base_path, 'data/processed/flights_processed.csv')  # NEW
        
        self.users = pd.read_csv(users_path)
        self.interactions = pd.read_csv(interactions_path)
        self.flights = pd.read_csv(flights_path)  # NEW: Load flights data
        
        # NEW: Compute reliability scores from delay rates
        self._compute_reliability_scores()
        
        logger.info(f"Loaded {len(self.users)} users, {len(self.interactions)} interactions")
        logger.info(f"Loaded {len(self.flights)} flights with reliability data")
    
    
    def _compute_reliability_scores(self) -> None:
        """
        Compute flight reliability from historical delay rates.
        
        Reliability = 1 - historical_delay_rate
        Higher reliability = more on-time flights
        """
        for _, flight in self.flights.iterrows():
            flight_id = flight['flight_id']
            delay_rate = flight.get('historical_delay_rate', 0)
            
            # Handle NaN delay rates
            if pd.isna(delay_rate):
                delay_rate = 0
            
            # Reliability = 1 - delay_rate (higher is better)
            reliability = float(max(0, min(1, 1 - delay_rate)))  # Clamp to [0, 1]
            self.reliability_scores[flight_id] = reliability
        
            # Reliability = 1 - delay_rate (higher is better)
            reliability = float(max(0, min(1, 1 - delay_rate)))  # Clamp to [0, 1]
            self.reliability_scores[flight_id] = reliability
        
        logger.info(f"Computed reliability scores for {len(self.reliability_scores)} flights")
    
    
    def infer_user_preferences_from_behavior(self, user_id: str) -> dict:
        """
        Infer user's actual preferences by analyzing their booking history.
        
        Learns: Does user actually book cheap/expensive flights? Fast/comfortable?
        
        Args:
            user_id: User identifier
        
        Returns:
            preferences: Dict with inferred 'price_pref' (0-100), 'time_pref' (0-100)
                        based on average booked flight attributes
        """
        try:
            user_bookings = self.interactions[self.interactions['user_id'] == user_id]
            
            if user_bookings.empty:
                return {'price_pref': 50, 'time_pref': 50}  # Neutral if no bookings
            
            # Get flight data for booked flights
            booked_flight_ids = user_bookings['flight_id'].values
            booked_flights = self.flights[self.flights['flight_id'].isin(booked_flight_ids)]
            
            if booked_flights.empty:
                return {'price_pref': 50, 'time_pref': 50}
            
            # Compute average attributes of booked flights
            avg_price = booked_flights['price'].mean()  # 0-1 normalized
            avg_duration = booked_flights['flight_duration_minutes'].mean()  # 0-1 normalized
            
            # Convert back to 0-100 scale
            inferred_price_pref = int(min(100, max(0, avg_price * 100)))
            inferred_time_pref = int(min(100, max(0, avg_duration * 100)))
            
            logger.debug(f"Inferred prefs for {user_id}: price={inferred_price_pref}, time={inferred_time_pref}")
            
            return {
                'price_pref': inferred_price_pref,
                'time_pref': inferred_time_pref
            }
        
        except Exception as e:
            logger.debug(f"Error inferring preferences for {user_id}: {e}")
            return {'price_pref': 50, 'time_pref': 50}  # Safe default
    
    
    def compute_preference_coherence(self, user_id: str, stated_prefs: dict) -> float:
        """
        Measure how well a user's stated preferences align with their actual behavior.
        
        High coherence → user's stated prefs are predictive of bookings
        Low coherence → user's stated prefs don't match their behavior
        
        Args:
            user_id: User identifier
            stated_prefs: Dict with stated 'price_pref', 'time_pref' (0-100)
        
        Returns:
            coherence_score: 0-1, higher = better alignment between stated and behavioral
        """
        try:
            # Infer preferences from actual booking history
            behavioral_prefs = self.infer_user_preferences_from_behavior(user_id)
            
            # Compute euclidean distance between stated and behavioral preferences
            price_diff = abs(stated_prefs.get('price_pref', 50) - behavioral_prefs['price_pref']) / 100.0
            time_diff = abs(stated_prefs.get('time_pref', 50) - behavioral_prefs['time_pref']) / 100.0
            
            distance = np.sqrt(price_diff**2 + time_diff**2)
            
            # Convert distance to coherence score using sigmoid-like function
            coherence = 1.0 / (1.0 + distance)  # Higher distance → lower coherence
            
            logger.debug(f"Coherence for {user_id}: {coherence:.2f} (distance={distance:.2f})")
            
            return float(coherence)
        
        except Exception as e:
            logger.debug(f"Error computing preference coherence: {e}")
            return 0.5  # Neutral default
    
    
    def blend_preferences(self, user_id: str, stated_prefs: dict) -> dict:
        """
        Blend stated and behavioral preferences based on coherence score.
        
        If user's stated prefs match their behavior (high coherence), trust stated more.
        If user's stated prefs don't match behavior (low coherence), trust behavior more.
        
        Args:
            user_id: User identifier
            stated_prefs: Dict with stated 'price_pref', 'time_pref' (0-100)
        
        Returns:
            blended_prefs: Dict with adjusted preferences
        """
        try:
            # Measure how reliable the stated preferences are
            coherence = self.compute_preference_coherence(user_id, stated_prefs)
            
            # Infer preferences from behavior
            behavioral_prefs = self.infer_user_preferences_from_behavior(user_id)
            
            # Blend: coherence weight on stated, (1-coherence) weight on behavioral
            # High coherence user: trust stated more
            # Low coherence user: trust behavior more
            weight_stated = max(0.4, coherence)  # Never fully ignore stated (min 40%)
            weight_behavioral = 1.0 - weight_stated
            
            blended = {
                'price_pref': int(
                    weight_stated * stated_prefs.get('price_pref', 50) +
                    weight_behavioral * behavioral_prefs['price_pref']
                ),
                'time_pref': int(
                    weight_stated * stated_prefs.get('time_pref', 50) +
                    weight_behavioral * behavioral_prefs['time_pref']
                ),
                'coherence': coherence  # Include coherence score in output
            }
            
            logger.info(
                f"Blended prefs for {user_id}: coherence={coherence:.2f}, "
                f"weight_stated={weight_stated:.2f}, stated={stated_prefs}, blended={blended}"
            )
            
            return blended
        
        except Exception as e:
            logger.debug(f"Error blending preferences: {e}")
            return stated_prefs  # Return stated as fallback
    
    
    def build_feature_matrix(self) -> np.ndarray:
        """
        Extract numeric features from users DataFrame.
        
        Returns:
            feature_matrix: (n_users, n_features) array of user features
        """
        # Select only numeric features (exclude user_id and other text columns)
        user_features = self.users.select_dtypes(include=[np.number, 'bool']).copy()
        
        # Convert bool to float
        for col in user_features.columns:
            if user_features[col].dtype == bool:
                user_features[col] = user_features[col].astype(float)
        
        # Handle NaN values by filling with column mean
        for col in user_features.columns:
            if user_features[col].isna().any():
                mean_val = user_features[col].mean()
                if np.isnan(mean_val):
                    # If all NaN, fill with 0
                    user_features[col].fillna(0, inplace=True)
                else:
                    user_features[col].fillna(mean_val, inplace=True)
        
        # Standardize features (important for cosine similarity)
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(user_features)
        
        # Final NaN check and replacement
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        logger.info(f"Built feature matrix: {feature_matrix.shape}")
        return feature_matrix
    
    
    def compute_similarity(self) -> None:
        """
        Compute cosine similarity between all users.
        
        Creates similarity_matrix: (n_users, n_users)
        """
        feature_matrix = self.build_feature_matrix()
        self.similarity_matrix = cosine_similarity(feature_matrix)
        
        logger.info(f"Computed similarity matrix: {self.similarity_matrix.shape}")
    
    
    def build_graph(self) -> None:
        """
        Build user similarity graph with similarity scores.
        
        Creates neighbors_with_scores: user_id → [(neighbor_id, similarity_score)]
        """
        self.neighbors_with_scores = {}
        self.user_index = {uid: i for i, uid in enumerate(self.users['user_id'].values)}
        
        for user_id, user_idx in self.user_index.items():
            # Get similarity scores for this user (excluding self)
            similarities = self.similarity_matrix[user_idx].copy()
            similarities[user_idx] = -np.inf  # Exclude self
            
            # Get top-K neighbors with their similarity scores
            top_k_indices = np.argsort(similarities)[-self.k_neighbors:][::-1]
            
            neighbors_scores = []
            for idx in top_k_indices:
                neighbor_id = self.users.iloc[idx]['user_id']
                similarity = float(similarities[idx])
                neighbors_scores.append((neighbor_id, similarity))
            
            self.neighbors_with_scores[user_id] = neighbors_scores
        
        logger.info(f"Built graph: {len(self.neighbors_with_scores)} users")
        logger.info(f"Neighbors per user: {self.k_neighbors}")
        logger.info(f"Using weighted scoring: Score = Σ(similarity × interaction)")
    
    
    def get_neighbors(self, user_id: str) -> List[Tuple[str, float]]:
        """
        Get top-K similar users with similarity scores.
        
        Args:
            user_id: User identifier
        
        Returns:
            List of (neighbor_id, similarity_score) tuples sorted by similarity
        """
        if user_id not in self.neighbors_with_scores:
            logger.warning(f"User {user_id} not found in graph")
            return []
        
        return self.neighbors_with_scores[user_id]
    
    
    def compute_preference_agreement(self, user_prefs: dict, neighbor_id: str, user_id: str) -> float:
        """
        Compute how much a neighbor's preferences align with the user's stated preferences.
        
        This estimates neighbor preference agreement from their interaction patterns.
        Users who book expensive/fast flights are assumed to prefer those attributes.
        
        Args:
            user_prefs: Dict with keys 'price_pref', 'time_pref', 'reliability_pref' (0-100)
            neighbor_id: The neighbor user to compare against
            user_id: The main user (for context)
        
        Returns:
            agreement_score: 0-1, higher = more preference alignment
        """
        try:
            # Get neighbor's bookings
            neighbor_bookings = self.interactions[
                self.interactions['user_id'] == neighbor_id
            ]
            
            if neighbor_bookings.empty:
                return 0.5  # No data, neutral agreement
            
            # Infer neighbor's preferences from their booking patterns
            # High avg price → inferred price preference is high (premium)
            # High avg duration → inferred time preference is high (comfort)
            avg_price = neighbor_bookings['flight_id'].apply(
                lambda fid: self.flights[self.flights['flight_id'] == fid]['price'].values[0] 
                if fid in self.flights['flight_id'].values else 0.5
            ).mean()
            
            avg_duration = neighbor_bookings['flight_id'].apply(
                lambda fid: self.flights[self.flights['flight_id'] == fid]['flight_duration_minutes'].values[0]
                if fid in self.flights['flight_id'].values else 0.5
            ).mean()
            
            # Convert inferred preferences to 0-100 scale
            inferred_price_pref = int(avg_price * 100)  # Flights are normalized 0-1
            inferred_time_pref = int(avg_duration * 100)
            
            # Compute euclidean distance between stated and inferred preferences
            user_price_pref = user_prefs.get('price_pref', 50)
            user_time_pref = user_prefs.get('time_pref', 50)
            
            price_diff = abs(user_price_pref - inferred_price_pref) / 100.0
            time_diff = abs(user_time_pref - inferred_time_pref) / 100.0
            
            # Convert distance to agreement (closer = higher agreement)
            # Using sigmoid-like function: agreement = 1 / (1 + distance)
            distance = np.sqrt(price_diff**2 + time_diff**2)
            agreement = 1.0 / (1.0 + distance)
            
            return float(agreement)
        
        except Exception as e:
            logger.debug(f"Error computing preference agreement: {e}")
            return 0.5  # Default if error
    
    
    def get_neighbors_with_preference_boost(self, user_id: str, user_prefs: dict) -> List[Tuple[str, float]]:
        """
        Get neighbors with similarity scores adjusted by preference agreement.
        
        Blends structural similarity (feature-based) with preference alignment.
        
        Args:
            user_id: User identifier
            user_prefs: Dict with 'price_pref', 'time_pref', 'reliability_pref' (0-100)
        
        Returns:
            List of (neighbor_id, adjusted_similarity) tuples
        """
        if user_id not in self.neighbors_with_scores:
            logger.warning(f"User {user_id} not found in graph")
            return []
        
        base_neighbors = self.neighbors_with_scores[user_id]
        adjusted_neighbors = []
        
        for neighbor_id, base_similarity in base_neighbors:
            # Compute preference agreement
            pref_agreement = self.compute_preference_agreement(user_prefs, neighbor_id, user_id)
            
            # Blend: 70% structural similarity + 30% preference agreement
            # This gives preference alignment meaningful but not dominant weight
            adjusted_similarity = 0.7 * base_similarity + 0.3 * pref_agreement
            
            adjusted_neighbors.append((neighbor_id, adjusted_similarity))
        
        # Re-sort by adjusted similarity
        adjusted_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Adjusted neighbors for {user_id} using preference agreement (70/30 blend)")
        
        return adjusted_neighbors
    
    
    def get_user_bookings(self, user_id: str) -> set:
        """
        Get all flights booked by a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            Set of flight_ids booked by user
        """
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        return set(user_interactions['flight_id'].values)
    
    
    def recommend(self, user_id: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Recommend flights using pure collaborative filtering (graph-based scoring).
        
        Algorithm:
        graph_score(flight) = Σ over neighbors:
                              similarity(user, neighbor) × interaction_weight(neighbor, flight)
        
        NOTE: Reliability is NOT applied here. It is applied separately in the endpoint
        as one component of a composite scoring function. This allows for proper
        weighting management and clear explainability.
        
        Where:
        - similarity: cosine similarity between feature vectors
        - interaction_weight: 1.0 for booking, 0.3 for view
        
        Args:
            user_id: User to recommend for
            top_n: Number of recommendations to return
        
        Returns:
            List of (flight_id, graph_score) tuples, sorted by score descending
        """
        neighbors_scores = self.get_neighbors(user_id)
        if not neighbors_scores:
            logger.warning(f"No neighbors found for user {user_id}")
            return []
        
        # Get user's own bookings (to exclude)
        user_bookings = self.get_user_bookings(user_id)
        
        # Compute graph-based weighted scores for flights (pure collaborative filtering)
        flight_scores = {}
        
        for neighbor_id, neighbor_similarity in neighbors_scores:
            # Get neighbor's interactions
            neighbor_interactions = self.interactions[
                self.interactions['user_id'] == neighbor_id
            ]
            
            for _, interaction in neighbor_interactions.iterrows():
                flight_id = interaction['flight_id']
                
                # Skip if user already booked this flight
                if flight_id in user_bookings:
                    continue
                
                # Get interaction weight based on type
                interaction_type = interaction.get('interaction_type', 'booking')
                weight = (
                    INTERACTION_WEIGHT_BOOKING 
                    if interaction_type == 'booking' 
                    else INTERACTION_WEIGHT_VIEW
                )
                
                # Score = similarity × interaction_weight
                contribution = neighbor_similarity * weight
                flight_scores[flight_id] = flight_scores.get(flight_id, 0) + contribution
        
        # Return pure graph scores (reliability applied separately in endpoint)
        recommendations = sorted(
            flight_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        logger.info(f"Generated {len(recommendations)} recommendations for {user_id}")
        logger.info(f"Score formula: Σ(similarity × interaction_weight)")
        
        return recommendations
    
    
    def recommend_with_preference_boost(self, user_id: str, user_prefs: dict, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Recommend flights using preference-adjusted collaborative filtering.
        
        Reweights neighbors based on how well their preferences align with the user's
        stated preferences, then computes graph scores using adjusted neighbor weights.
        
        Algorithm:
        adjusted_similarity = 0.7 * structural_similarity + 0.3 * preference_agreement
        graph_score(flight) = Σ over neighbors:
                              adjusted_similarity(user, neighbor) × interaction_weight(neighbor, flight)
        
        Args:
            user_id: User to recommend for
            user_prefs: Dict with 'price_pref', 'time_pref', 'reliability_pref' (0-100)
            top_n: Number of recommendations to return
        
        Returns:
            List of (flight_id, graph_score) tuples, sorted by score descending
        """
        # Get neighbors adjusted by preference agreement
        neighbors_scores = self.get_neighbors_with_preference_boost(user_id, user_prefs)
        if not neighbors_scores:
            logger.warning(f"No neighbors found for user {user_id}")
            return []
        
        # Get user's own bookings (to exclude)
        user_bookings = self.get_user_bookings(user_id)
        
        # Compute graph-based weighted scores using adjusted neighbors
        flight_scores = {}
        
        for neighbor_id, adjusted_similarity in neighbors_scores:
            # Get neighbor's interactions
            neighbor_interactions = self.interactions[
                self.interactions['user_id'] == neighbor_id
            ]
            
            for _, interaction in neighbor_interactions.iterrows():
                flight_id = interaction['flight_id']
                
                # Skip if user already booked this flight
                if flight_id in user_bookings:
                    continue
                
                # Get interaction weight based on type
                interaction_type = interaction.get('interaction_type', 'booking')
                weight = (
                    INTERACTION_WEIGHT_BOOKING 
                    if interaction_type == 'booking' 
                    else INTERACTION_WEIGHT_VIEW
                )
                
                # Score = adjusted_similarity × interaction_weight
                contribution = adjusted_similarity * weight
                flight_scores[flight_id] = flight_scores.get(flight_id, 0) + contribution
        
        # Return graph scores (preference-adjusted)
        recommendations = sorted(
            flight_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        logger.info(f"Generated {len(recommendations)} preference-boosted recommendations for {user_id}")
        logger.info(f"Score formula: Σ(adjusted_similarity × interaction_weight) where adjusted=0.7*structural + 0.3*pref_agreement")
        
        return recommendations
    
    
    def build_pipeline(self) -> None:
        """
        Run complete pipeline: load data → compute similarity → build graph.
        """
        logger.info("=" * 80)
        logger.info("STARTING USER SIMILARITY GRAPH PIPELINE")
        logger.info("=" * 80)
        
        self.load_data()
        self.compute_similarity()
        self.build_graph()
        
        logger.info("=" * 80)
        logger.info("✅ GRAPH PIPELINE COMPLETE")
        logger.info("=" * 80)


def get_recommender() -> UserSimilarityGraph:
    """
    Convenience function to create, initialize, and return a recommender.
    
    Returns:
        UserSimilarityGraph instance (pipeline run)
    """
    recommender = UserSimilarityGraph(k_neighbors=K_NEIGHBORS)
    recommender.build_pipeline()
    return recommender


if __name__ == '__main__':
    # Build graph and test recommendations
    recommender = get_recommender()
    
    print("\n" + "=" * 80)
    print("WEIGHTED SIMILARITY RECOMMENDATIONS")
    print("=" * 80)
    
    # Try recommendations for first 5 users
    sample_users = recommender.users['user_id'].values[:5]
    
    for user_id in sample_users:
        recommendations = recommender.recommend(user_id, top_n=5)
        neighbors = recommender.get_neighbors(user_id)
        
        print(f"\n👤 User {user_id}:")
        print(f"   Neighbors (similarity):")
        for neighbor_id, sim in neighbors[:3]:
            print(f"     - {neighbor_id}: {sim:.3f}")
        
        print(f"   📍 Booked: {list(recommender.get_user_bookings(user_id))[:3]}...")
        print(f"   ✈️  Recommendations (weighted score):")
        for flight_id, score in recommendations:
            print(f"     - {flight_id}: {score:.3f}")
    
    print("\n" + "=" * 80)
    print("✅ Weighted Graph Recommender ready for use!")
    print("=" * 80)
