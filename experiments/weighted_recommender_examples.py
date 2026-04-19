"""
Weighted Graph Recommender — Usage Examples

Demonstrates how to use the improved weighted similarity scoring
for flight recommendations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from graph_recommender import UserSimilarityGraph


# ============================================================================
# Example 1: Basic Usage
# ============================================================================

def example_basic_usage():
    """Simplest way to get recommendations."""
    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC USAGE")
    print("="*80)
    
    # Initialize and build
    recommender = UserSimilarityGraph(k_neighbors=5)
    recommender.build_pipeline()
    
    # Get recommendations for a user
    user_id = 'U00001'
    recommendations = recommender.recommend(user_id, top_n=5)
    
    print(f"\nRecommendations for {user_id}:")
    for flight_id, score in recommendations:
        print(f"  {flight_id}: {score:.3f}")


# ============================================================================
# Example 2: Understanding the Scoring
# ============================================================================

def example_understand_scoring():
    """Show how weighted scores are computed."""
    print("\n" + "="*80)
    print("EXAMPLE 2: UNDERSTANDING WEIGHTED SCORES")
    print("="*80)
    
    recommender = UserSimilarityGraph(k_neighbors=5)
    recommender.build_pipeline()
    
    user_id = 'U00001'
    
    # Get neighbors with similarity scores
    neighbors = recommender.get_neighbors(user_id)
    print(f"\nTop-5 Similar Users for {user_id}:")
    print("  User ID  │  Similarity")
    print("  ─────────┼────────────")
    for neighbor_id, sim_score in neighbors:
        print(f"  {neighbor_id}  │   {sim_score:.4f}")
    
    # Get recommendations
    recommendations = recommender.recommend(user_id, top_n=3)
    print(f"\nTop-3 Weighted Recommendations:")
    print("  Flight ID │  Weighted Score")
    print("  ──────────┼─────────────────")
    for flight_id, score in recommendations:
        print(f"  {flight_id}    │    {score:.4f}")
    
    # Show scoring formula
    print(f"\nScoring Formula:")
    print(f"  Score(flight) = Σ over neighbors:")
    print(f"                  similarity(user, neighbor) ×")
    print(f"                  interaction_weight(neighbor, flight)")
    print(f"\n  Where:")
    print(f"    - similarity ∈ [0, 1]")
    print(f"    - interaction_weight = 1.0 for booking")
    print(f"                         = 0.3 for view/search")


# ============================================================================
# Example 3: Compare Neighbors per User
# ============================================================================

def example_neighbor_comparison():
    """Show neighbors for multiple users."""
    print("\n" + "="*80)
    print("EXAMPLE 3: NEIGHBOR COMPARISON")
    print("="*80)
    
    recommender = UserSimilarityGraph(k_neighbors=5)
    recommender.build_pipeline()
    
    users = ['U00001', 'U00002', 'U00003']
    
    for user_id in users:
        neighbors = recommender.get_neighbors(user_id)
        avg_similarity = sum(sim for _, sim in neighbors) / len(neighbors)
        
        print(f"\n{user_id}:")
        print(f"  Avg Neighbor Similarity: {avg_similarity:.4f}")
        print(f"  Neighbors:")
        for neighbor_id, sim in neighbors:
            print(f"    - {neighbor_id}: {sim:.4f}")


# ============================================================================
# Example 4: Batch Recommendations
# ============================================================================

def example_batch_recommendations():
    """Get recommendations for multiple users efficiently."""
    print("\n" + "="*80)
    print("EXAMPLE 4: BATCH RECOMMENDATIONS")
    print("="*80)
    
    recommender = UserSimilarityGraph(k_neighbors=5)
    recommender.build_pipeline()
    
    # Get recommendations for first 5 users
    users = ['U00001', 'U00002', 'U00003', 'U00004', 'U00005']
    
    print(f"\nGetting recommendations for {len(users)} users...")
    
    all_recommendations = {}
    for user_id in users:
        recs = recommender.recommend(user_id, top_n=3)
        all_recommendations[user_id] = recs
    
    # Display results
    for user_id, recommendations in all_recommendations.items():
        print(f"\n{user_id}:")
        for flight_id, score in recommendations:
            print(f"  {flight_id}: {score:.3f}")


# ============================================================================
# Example 5: Weighted Score Breakdown
# ============================================================================

def example_score_breakdown():
    """Manual calculation of a single recommendation score."""
    print("\n" + "="*80)
    print("EXAMPLE 5: WEIGHTED SCORE BREAKDOWN")
    print("="*80)
    
    recommender = UserSimilarityGraph(k_neighbors=3)  # Use k=3 for clarity
    recommender.build_pipeline()
    
    user_id = 'U00001'
    
    # Get neighbors
    neighbors = recommender.get_neighbors(user_id)[:3]
    
    print(f"\nUser: {user_id}")
    print(f"Top-3 Neighbors (with similarity):")
    for neighbor_id, sim in neighbors:
        print(f"  {neighbor_id}: similarity = {sim:.4f}")
    
    # Get first recommendation
    recommendations = recommender.recommend(user_id, top_n=1)
    if recommendations:
        rec_flight_id, rec_score = recommendations[0]
        
        print(f"\nTop Recommendation: {rec_flight_id}")
        print(f"Weighted Score: {rec_score:.4f}")
        print(f"\nHow was this calculated?")
        print(f"  Score = Σ(similarity × interaction_weight)")
        print(f"\n  Contributions from neighbors:")
        
        # Manual calculation
        total = 0
        for neighbor_id, sim in neighbors:
            neighbor_interactions = recommender.interactions[
                recommender.interactions['user_id'] == neighbor_id
            ]
            
            for _, interaction in neighbor_interactions.iterrows():
                if interaction['flight_id'] == rec_flight_id:
                    weight = 1.0 if interaction['interaction_type'] == 'booking' else 0.3
                    contribution = sim * weight
                    interaction_type = interaction['interaction_type']
                    print(f"    {neighbor_id}: {sim:.4f} × {weight} ({interaction_type}) = {contribution:.4f}")
                    total += contribution
        
        print(f"\n  Total Score: {total:.4f}")
        assert abs(total - rec_score) < 0.001, "Score mismatch!"


# ============================================================================
# Example 6: Parameter Tuning
# ============================================================================

def example_parameter_tuning():
    """Show impact of k_neighbors parameter."""
    print("\n" + "="*80)
    print("EXAMPLE 6: PARAMETER TUNING (k_neighbors)")
    print("="*80)
    
    user_id = 'U00001'
    
    for k in [3, 5, 10]:
        print(f"\nUsing k_neighbors={k}:")
        
        recommender = UserSimilarityGraph(k_neighbors=k)
        recommender.build_pipeline()
        
        recommendations = recommender.recommend(user_id, top_n=3)
        neighbors = recommender.get_neighbors(user_id)
        
        avg_sim = sum(sim for _, sim in neighbors) / len(neighbors)
        
        print(f"  Avg neighbor similarity: {avg_sim:.4f}")
        print(f"  Top recommendation score: {recommendations[0][1]:.4f}")
        print(f"  Recommendations: {[f[0] for f in recommendations]}")
    
    print(f"\n→ Larger k: more diverse but potentially noisier")
    print(f"→ Smaller k: cleaner signal but narrower coverage")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    examples = [
        ('1', example_basic_usage),
        ('2', example_understand_scoring),
        ('3', example_neighbor_comparison),
        ('4', example_batch_recommendations),
        ('5', example_score_breakdown),
        ('6', example_parameter_tuning),
    ]
    
    print("\n" + "="*80)
    print("WEIGHTED GRAPH RECOMMENDER — USAGE EXAMPLES")
    print("="*80)
    
    # Run all examples
    for num, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n⚠️  Example {num} error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*80)
