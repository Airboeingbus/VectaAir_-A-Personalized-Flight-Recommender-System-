"""
User Clustering Module for Flight Recommendation System (with SVD Reduction)

Combines user features with interaction patterns to create meaningful user clusters.
Uses TruncatedSVD to reduce sparse interaction matrix before clustering.

Pipeline:
1. Load preprocessed users, flights, and interactions
2. Build user-flight interaction matrix (rows: users, cols: flights)
3. Apply TruncatedSVD to reduce interaction matrix dimensionality
4. Combine user features with reduced interaction embedding (weighted 0.5)
5. Apply K-means clustering (with silhouette score evaluation)
6. Output cluster assignments and metrics to CSV

Output:
- data/processed/user_clusters.csv: user_id, cluster_id
- data/processed/clustering_metrics.csv: k, silhouette_score, cluster_distribution
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict, Any, List

# Configuration
K_CLUSTERS = 5  # Default number of user clusters
SVD_COMPONENTS = 30  # Reduce interaction matrix to 30 dimensions
INTERACTION_WEIGHT = 0.5  # Weight for interaction features (prevent dominance)
RANDOM_STATE = 42

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed data and interactions.
    
    Returns:
        users: DataFrame with user features
        flights: DataFrame with flight features
        interactions: DataFrame with user-flight-interaction triplets
    """
    users = pd.read_csv('../data/processed/users_processed.csv')
    flights = pd.read_csv('../data/processed/flights_processed.csv')
    interactions = pd.read_csv('../data/processed/interactions.csv')
    
    logger.info(f"Loaded {len(users)} users, {len(flights)} flights, {len(interactions)} interactions")
    return users, flights, interactions


def build_interaction_matrix(
    users: pd.DataFrame,
    flights: pd.DataFrame,
    interactions: pd.DataFrame
) -> pd.DataFrame:
    """
    Build user-flight interaction matrix.
    
    Args:
        users: User DataFrame
        flights: Flight DataFrame
        interactions: Interaction DataFrame (user_id, flight_id, interaction)
    
    Returns:
        interaction_matrix: DataFrame (rows: user_id, cols: flight_id, values: interaction)
                           Missing interactions = 0
    """
    # Create empty matrix (users × flights)
    user_ids = sorted(users['user_id'].values)
    flight_ids = sorted(flights['flight_id'].values)
    matrix = pd.DataFrame(
        0.0,
        index=user_ids,
        columns=flight_ids
    )
    
    # Fill matrix with interaction values
    for _, row in interactions.iterrows():
        user_id = row['user_id']
        flight_id = row['flight_id']
        interaction = row['interaction']
        
        if user_id in matrix.index and flight_id in matrix.columns:
            matrix.loc[user_id, flight_id] = interaction
    
    logger.info(f"Built interaction matrix: ({len(matrix)}, {len(matrix.columns)})")
    logger.info(f"  Sparsity: {(matrix == 0).sum().sum() / (len(matrix) * len(matrix.columns)) * 100:.1f}%")
    
    return matrix


def reduce_interaction_matrix(
    interaction_matrix: pd.DataFrame,
    n_components: int = SVD_COMPONENTS
) -> np.ndarray:
    """
    Apply TruncatedSVD to reduce sparse interaction matrix dimensionality.
    
    Args:
        interaction_matrix: User × flight sparse matrix
        n_components: Number of SVD components (target dimensions)
    
    Returns:
        reduced_embedding: User embedding (n_users × n_components)
    """
    # Apply SVD
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    reduced = svd.fit_transform(interaction_matrix.values)
    
    explained_var = svd.explained_variance_ratio_.sum()
    logger.info(f"SVD reduction: {interaction_matrix.shape[1]} → {n_components} dimensions")
    logger.info(f"  Explained variance: {explained_var*100:.1f}%")
    
    return reduced


def combine_features(
    users: pd.DataFrame,
    interaction_reduced: np.ndarray,
    weight_interactions: float = INTERACTION_WEIGHT
) -> pd.DataFrame:
    """
    Combine user features with reduced interaction embedding.
    
    User representation = [user_features, weight × interaction_embedding]
    
    Args:
        users: User features DataFrame
        interaction_reduced: Reduced interaction embedding (from SVD)
        weight_interactions: Weight for interaction features (default 0.5)
    
    Returns:
        combined: DataFrame with user_id index, features + weighted interactions
    """
    # Prepare user features (drop non-numeric columns except user_id)
    users_indexed = users.set_index('user_id')
    if 'user_id' in users_indexed.columns:
        users_indexed = users_indexed.drop(columns=['user_id'])
    
    # Select only numeric columns
    users_numeric = users_indexed.select_dtypes(include=[np.number, 'bool']).copy()
    for col in users_numeric.columns:
        if users_numeric[col].dtype == bool:
            users_numeric[col] = users_numeric[col].astype(float)
    
    # Weight and combine
    interaction_weighted = interaction_reduced * weight_interactions
    combined = pd.concat(
        [users_numeric.reset_index(drop=True),
         pd.DataFrame(interaction_weighted, columns=[f"svd_{i}" for i in range(interaction_weighted.shape[1])])],
        axis=1
    )
    combined.index = users_indexed.index
    
    logger.info(f"Combined features: shape = {combined.shape}")
    logger.info(f"  User features: {users_numeric.shape[1]}, Reduced interaction: {interaction_weighted.shape[1]}")
    
    return combined


def apply_clustering(
    combined_features: pd.DataFrame,
    k: int = K_CLUSTERS
) -> Tuple[np.ndarray, KMeans, float]:
    """
    Apply K-means clustering with silhouette score evaluation.
    
    Args:
        combined_features: Combined user features + reduced interactions
        k: Number of clusters
    
    Returns:
        cluster_labels: Array of cluster assignments
        kmeans_model: Fitted KMeans model
        silhouette_avg: Silhouette score (-1 to 1, higher is better)
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Compute silhouette score
    silhouette_avg = silhouette_score(scaled_features, cluster_labels)
    
    logger.info(f"K-means clustering complete (k={k})")
    logger.info(f"  Silhouette score: {silhouette_avg:.4f}")
    
    return cluster_labels, kmeans, silhouette_avg


def save_clusters(
    users: pd.DataFrame,
    cluster_labels: np.ndarray,
    silhouette_score: float,
    k: int
) -> pd.DataFrame:
    """
    Save cluster assignments and metrics to CSV.
    
    Args:
        users: User DataFrame
        cluster_labels: Cluster assignment array
        silhouette_score: Silhouette score for this clustering
        k: Number of clusters used
    
    Returns:
        result: DataFrame with user_id and cluster_id
    """
    result = pd.DataFrame({
        'user_id': users['user_id'],
        'cluster_id': cluster_labels
    })
    
    # Save to file
    os.makedirs('../data/processed', exist_ok=True)
    result.to_csv('../data/processed/user_clusters.csv', index=False)
    logger.info(f"Saved clusters to user_clusters.csv")
    
    # Print cluster statistics
    cluster_counts = result['cluster_id'].value_counts().sort_index()
    logger.info("Cluster distribution:")
    for cluster_id, count in cluster_counts.items():
        logger.info(f"  Cluster {cluster_id}: {count} users ({count/len(result)*100:.1f}%)")
    
    logger.info(f"Silhouette score (k={k}): {silhouette_score:.4f}")
    
    return result


def evaluate_multiple_k(
    combined_features: pd.DataFrame,
    k_values: List[int] = [3, 5, 8]
) -> Dict[int, float]:
    """
    Evaluate clustering quality for multiple K values using silhouette score.
    
    Args:
        combined_features: Combined user features + reduced interactions
        k_values: List of K values to test
    
    Returns:
        metrics: Dict mapping k → silhouette_score
    """
    metrics = {}
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)
    
    logger.info(f"\nEvaluating K values: {k_values}")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        metrics[k] = score
        logger.info(f"  k={k}: silhouette_score = {score:.4f}")
    
    return metrics


def cluster_pipeline(
    k: int = K_CLUSTERS,
    test_multiple_k: bool = True,
    k_values: List[int] = [3, 5, 8]
) -> Dict[str, Any]:
    """
    Complete clustering pipeline with dimensionality reduction.
    
    Steps:
    1. Load users, flights, interactions
    2. Build user-flight interaction matrix
    3. Apply SVD reduction
    4. Combine features with weighted interactions
    5. Apply K-means clustering (with silhouette score)
    6. Optionally test multiple K values
    7. Save best result
    
    Args:
        k: Primary number of clusters
        test_multiple_k: Whether to evaluate multiple K values
        k_values: List of K values to test
    
    Returns:
        Dictionary with clustering results and metrics
    """
    logger.info("=" * 80)
    logger.info("STARTING USER CLUSTERING PIPELINE (WITH SVD REDUCTION)")
    logger.info("=" * 80)
    
    # Step 1: Load
    users, flights, interactions = load_data()
    
    # Step 2: Build interaction matrix
    interaction_matrix = build_interaction_matrix(users, flights, interactions)
    
    # Step 3: Apply SVD reduction
    interaction_reduced = reduce_interaction_matrix(interaction_matrix, n_components=SVD_COMPONENTS)
    
    # Step 4: Combine features
    combined = combine_features(users, interaction_reduced, weight_interactions=INTERACTION_WEIGHT)
    
    # Step 5: Evaluate multiple K values
    metrics = {}
    if test_multiple_k:
        metrics = evaluate_multiple_k(combined, k_values=k_values)
        best_k = max(metrics, key=metrics.get)
        logger.info(f"\nBest K: {best_k} (silhouette={metrics[best_k]:.4f})")
        k = best_k
    
    # Step 6: Apply clustering with best K
    clusters, kmeans, sil_score = apply_clustering(combined, k=k)
    
    # Step 7: Save
    result = save_clusters(users, clusters, sil_score, k)
    
    logger.info("=" * 80)
    logger.info("✅ CLUSTERING PIPELINE COMPLETE")
    logger.info("=" * 80)
    
    return {
        'users': users,
        'clusters': result,
        'interaction_matrix': interaction_matrix,
        'interaction_reduced': interaction_reduced,
        'combined_features': combined,
        'kmeans_model': kmeans,
        'silhouette_score': sil_score,
        'k': k,
        'metrics_all_k': metrics
    }


if __name__ == '__main__':
    # Run clustering pipeline with multiple K evaluation
    output = cluster_pipeline(k=K_CLUSTERS, test_multiple_k=True, k_values=[3, 5, 8])
    print(f"\n✅ Clustering complete!")
    print(f"Output saved to: data/processed/user_clusters.csv")
    print(f"Best K: {output['k']}, Silhouette Score: {output['silhouette_score']:.4f}")
