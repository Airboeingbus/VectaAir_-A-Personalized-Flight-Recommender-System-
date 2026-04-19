"""
Example usage patterns for user clustering module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from clustering import cluster_pipeline, load_data, build_interaction_matrix, combine_features
import pandas as pd


def example_full_pipeline():
    """
    Example 1: Full clustering pipeline (recommended)
    
    Loads data, builds interaction matrix, applies clustering, saves results.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Full Clustering Pipeline")
    print("="*80 + "\n")
    
    # Run complete pipeline with 5 clusters
    output = cluster_pipeline(k=5)
    
    # Access results
    clusters = output['clusters']
    print(f"\nCluster assignments shape: {clusters.shape}")
    print(f"\nFirst 10 users and their clusters:\n{clusters.head(10)}")
    
    # Inspect cluster distribution
    print(f"\nCluster distribution:\n{clusters['cluster_id'].value_counts().sort_index()}")
    
    return output


def example_step_by_step():
    """
    Example 2: Step-by-step clustering process
    
    Demonstrates each step individually for understanding/debugging.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Step-by-Step Process")
    print("="*80 + "\n")
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Step 1: Load data
    print("[1/5] Loading data...")
    users, flights, interactions = load_data()
    print(f"  Users: {len(users)}, Flights: {len(flights)}, Interactions: {len(interactions)}")
    
    # Step 2: Build interaction matrix
    print("\n[2/5] Building interaction matrix...")
    interaction_matrix = build_interaction_matrix(users, flights, interactions)
    print(f"  Shape: {interaction_matrix.shape}")
    print(f"  Sample row (user {users.iloc[0]['user_id']}):\n    {interaction_matrix.iloc[0].head()}")
    
    # Step 3: Combine features
    print("\n[3/5] Combining user features + interactions...")
    combined = combine_features(users, interaction_matrix)
    print(f"  Combined shape: {combined.shape}")
    print(f"  User feature columns: {list(users.columns)[:5]}...")
    print(f"  Interaction vector columns: {list(interaction_matrix.columns)[:3]}...")
    
    # Step 4: Apply clustering
    print("\n[4/5] Applying K-means clustering...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    print(f"  Clusters assigned: {len(clusters)}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    
    # Step 5: Save results
    print("\n[5/5] Saving results...")
    result = pd.DataFrame({'user_id': users['user_id'], 'cluster_id': clusters})
    result.to_csv('../data/processed/user_clusters.csv', index=False)
    print(f"  Saved to user_clusters.csv")
    
    return result


def example_inspect_clusters():
    """
    Example 3: Analyze cluster characteristics
    
    Load results and inspect what makes each cluster distinct.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Analyze Cluster Characteristics")
    print("="*80 + "\n")
    
    # Load everything
    output = cluster_pipeline(k=5)
    users = output['users']
    clusters = output['clusters']
    interaction_matrix = output['interaction_matrix']
    
    # Merge to analyze
    analysis = clusters.copy()
    
    # Add user features
    for col in ['age', 'gender_Female', 'gender_Male']:
        if col in users.columns:
            user_features_df = pd.DataFrame({
                'user_id': users['user_id'],
                col: users[col]
            })
            analysis = analysis.merge(user_features_df, on='user_id', how='left')
    
    # Cluster-wise statistics
    print("Cluster Statistics:\n")
    for cluster_id in sorted(analysis['cluster_id'].unique()):
        cluster_users = analysis[analysis['cluster_id'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_users)} users):")
        
        if 'age' in cluster_users.columns:
            print(f"  Avg age: {cluster_users['age'].mean():.1f}")
        if 'gender_Female' in cluster_users.columns:
            female_pct = cluster_users['gender_Female'].mean() * 100
            print(f"  Female: {female_pct:.1f}%")
        
        # Interaction stats
        cluster_interaction = interaction_matrix.loc[cluster_users['user_id']]
        avg_interactions = (cluster_interaction > 0).sum().mean()
        print(f"  Avg flights booked: {avg_interactions:.1f}")
    
    return analysis


if __name__ == '__main__':
    print("\n🚀 USER CLUSTERING EXAMPLES\n")
    
    # Run examples
    output1 = example_full_pipeline()
    result2 = example_step_by_step()
    analysis = example_inspect_clusters()
    
    print("\n✅ All examples complete!")
