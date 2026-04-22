"""
Evaluation Script: Precision@5 and Recall@5 Metrics

Splits interactions into train/test sets and evaluates the hybrid recommender
using standard information retrieval metrics.

Usage:
    python scripts/evaluate_recommender.py
"""

import os
import sys
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from graph_recommender import UserSimilarityGraph


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_and_split_interactions_random(test_size=0.2, random_state=SEED):
    """
    Load interactions and split into train/test sets per-user.
    Ensures each user has both training and test interactions.
    
    Args:
        test_size: Fraction of each user's interactions for test set
        random_state: Seed for reproducibility
    
    Returns:
        train_interactions, test_interactions (DataFrames)
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    interactions_path = os.path.join(base_path, 'data/processed/interactions.csv')
    
    interactions = pd.read_csv(interactions_path)
    
    # Split per user to ensure train/test overlap
    train_list = []
    test_list = []
    
    for user_id in sorted(interactions['user_id'].unique()):
        user_interactions = interactions[interactions['user_id'] == user_id]
        
        # Only split if user has at least 3 interactions
        if len(user_interactions) >= 3:
            user_train, user_test = train_test_split(
                user_interactions,
                test_size=test_size,
                random_state=random_state
            )
            train_list.append(user_train)
            test_list.append(user_test)
    
    train = pd.concat(train_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)
    
    print(f"Total interactions: {len(interactions)}")
    print(f"Training interactions: {len(train)}")
    print(f"Test interactions: {len(test)}")
    print(f"Users with both train+test: {len(train_list)}")
    print("Split strategy: random per-user split")
    
    return train, test


def load_and_split_interactions_leave_one_out(random_state=SEED):
    """
    Load interactions and build deterministic leave-one-out split per user.

    Each user with >=2 interactions contributes exactly one held-out test
    interaction; remaining interactions are kept for training.

    Args:
        random_state: Seed for deterministic held-out selection

    Returns:
        train_interactions, test_interactions (DataFrames)
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    interactions_path = os.path.join(base_path, 'data/processed/interactions.csv')

    interactions = pd.read_csv(interactions_path)

    train_list = []
    test_list = []

    for user_id in sorted(interactions['user_id'].unique()):
        user_interactions = interactions[interactions['user_id'] == user_id]

        # Leave-one-out requires at least 2 interactions
        if len(user_interactions) >= 2:
            user_test = user_interactions.sample(n=1, random_state=random_state)
            user_train = user_interactions.drop(user_test.index)
            train_list.append(user_train)
            test_list.append(user_test)

    train = pd.concat(train_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)

    print(f"Total interactions: {len(interactions)}")
    print(f"Training interactions: {len(train)}")
    print(f"Test interactions: {len(test)}")
    print(f"Users with both train+test: {len(train_list)}")
    print("Split strategy: leave-one-out per-user split")

    return train, test


def train_recommender_on_subset(train_interactions):
    """
    Train recommender using only training interactions.
    
    Args:
        train_interactions: DataFrame with columns [user_id, flight_id, interaction]
    
    Returns:
        Trained UserSimilarityGraph instance
    """
    # Initialize recommender
    recommender = UserSimilarityGraph(k_neighbors=5)
    
    # Load full data (users, flights) but use only training interactions
    recommender.load_data()
    
    # Replace interactions with training subset
    recommender.interactions = train_interactions.reset_index(drop=True)
    
    # Build graph from training data
    recommender.build_feature_matrix()
    recommender.compute_similarity()
    recommender.build_graph()
    
    print(f"Recommender trained on {len(train_interactions)} interactions")
    return recommender


def get_test_user_interactions(test_interactions, min_interactions=1):
    """
    Get test interactions grouped by user.
    Filter out users with insufficient interactions for meaningful evaluation.
    
    Args:
        test_interactions: DataFrame with test interactions
        min_interactions: Minimum interactions per user (default 1 since we split per-user)
    
    Returns:
        Dictionary mapping user_id → list of flight_ids (ground truth)
    """
    user_flights = {}
    
    for user_id in sorted(test_interactions['user_id'].unique()):
        flights = test_interactions[test_interactions['user_id'] == user_id]['flight_id'].tolist()
        
        if len(flights) >= min_interactions:
            user_flights[user_id] = flights
    
    print(f"Users with ≥{min_interactions} test interactions: {len(user_flights)}")
    return user_flights


def compute_precision_recall_at_k(recommendations, ground_truth, k=5):
    """
    Compute precision@k and recall@k.
    
    Args:
        recommendations: List of (flight_id, score) tuples (ranked by score)
        ground_truth: Set of relevant flight_ids
        k: Cutoff for precision/recall
    
    Returns:
        Tuple of (precision@k, recall@k)
    """
    # Get top-k recommended flights
    top_k = [flight_id for flight_id, _ in recommendations[:k]]
    
    # Convert ground truth to set
    gt_set = set(ground_truth)
    top_k_set = set(top_k)
    
    # Compute hits
    hits = len(gt_set & top_k_set)
    
    # Precision: hits / k
    precision = hits / k if k > 0 else 0
    
    # Recall: hits / len(ground_truth)
    recall = hits / len(gt_set) if len(gt_set) > 0 else 0
    
    return precision, recall


def evaluate_recommender(recommender, test_user_interactions, top_n=5):
    """
    Evaluate recommender on test set.
    
    Args:
        recommender: Trained UserSimilarityGraph instance
        test_user_interactions: Dict mapping user_id → list of ground truth flights
        top_n: Number of recommendations per user
    
    Returns:
        Dict with metrics
    """
    precisions = []
    recalls = []
    evaluated_users = 0
    
    for user_id, ground_truth in test_user_interactions.items():
        try:
            # Get recommendations
            recommendations = recommender.recommend(user_id, top_n=top_n)
            
            if recommendations:
                # Compute metrics
                p, r = compute_precision_recall_at_k(recommendations, ground_truth, k=top_n)
                precisions.append(p)
                recalls.append(r)
                evaluated_users += 1
        
        except Exception as e:
            # User may not be in training data (cold start)
            continue
    
    # Compute averages
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    
    print(f"\nEvaluated {evaluated_users} users")
    
    return {
        'precision@5': avg_precision,
        'recall@5': avg_recall,
        'num_users': evaluated_users,
        'precisions': precisions,
        'recalls': recalls
    }


def print_results(metrics):
    """Pretty print evaluation results."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Metric              | Value")
    print("-"*70)
    print(f"Precision@5         | {metrics['precision@5']:.4f}")
    print(f"Recall@5            | {metrics['recall@5']:.4f}")
    print(f"Users Evaluated     | {metrics['num_users']}")
    print("="*70)
    
    # Additional statistics
    if metrics['precisions']:
        print(f"\nPrecision@5 Statistics:")
        print(f"  Min:     {min(metrics['precisions']):.4f}")
        print(f"  Max:     {max(metrics['precisions']):.4f}")
        print(f"  Median:  {np.median(metrics['precisions']):.4f}")
        print(f"  Std:     {np.std(metrics['precisions']):.4f}")
    
    if metrics['recalls']:
        print(f"\nRecall@5 Statistics:")
        print(f"  Min:     {min(metrics['recalls']):.4f}")
        print(f"  Max:     {max(metrics['recalls']):.4f}")
        print(f"  Median:  {np.median(metrics['recalls']):.4f}")
        print(f"  Std:     {np.std(metrics['recalls']):.4f}")


def main():
    """Run full evaluation pipeline."""
    print("\n" + "="*70)
    print("HYBRID RECOMMENDER EVALUATION")
    print("="*70 + "\n")
    
    # Load and split data
    print("1. Loading and splitting data...")
    train_interactions, test_interactions = load_and_split_interactions_random(test_size=0.2)

    # Validate held-out user count and switch to leave-one-out if needed
    test_user_interactions = get_test_user_interactions(test_interactions)
    test_user_count = len(test_user_interactions)
    print(f"Test users with held-out interactions: {test_user_count}")

    if test_user_count < 100:
        print("Held-out test users < 100. Switching to leave-one-out evaluation...")
        train_interactions, test_interactions = load_and_split_interactions_leave_one_out()
        test_user_interactions = get_test_user_interactions(test_interactions)
        test_user_count = len(test_user_interactions)
        print(f"Test users with held-out interactions: {test_user_count}")
    
    # Train recommender
    print("\n2. Training recommender on training set...")
    recommender = train_recommender_on_subset(train_interactions)
    
    # Get test ground truth
    print("\n3. Preparing test set...")
    # Already prepared above to validate held-out user count
    
    # Evaluate
    print("\n4. Evaluating on test set...")
    metrics = evaluate_recommender(recommender, test_user_interactions, top_n=5)
    print(f"Test users with held-out interactions: {test_user_count}")
    
    # Print results
    print_results(metrics)
    
    # Return metrics for potential further use
    return metrics


if __name__ == '__main__':
    metrics = main()
    
    # Output metrics in a format that can be parsed
    print("\n" + "="*70)
    print(f"SUMMARY FOR README:")
    print(f"precision@5={metrics['precision@5']:.4f}, recall@5={metrics['recall@5']:.4f}")
    print("="*70)
