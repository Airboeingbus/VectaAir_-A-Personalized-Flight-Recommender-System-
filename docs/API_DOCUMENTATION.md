# Flight Recommender System - API Documentation

## Overview

The Flight Recommender System provides multiple recommendation engines to suggest flights to users based on their preferences and behavior. This documentation covers all public APIs and their usage patterns.

---

## Table of Contents

1. [Content-Based Recommender](#content-based-recommender)
2. [Collaborative Recommender](#collaborative-recommender)
3. [Graph-Based Recommender](#graph-based-recommender)
4. [Recommender Ensemble](#recommender-ensemble)
5. [Data Models](#data-models)
6. [Usage Examples](#usage-examples)

---

## Content-Based Recommender

### Overview

The content-based recommender suggests flights similar to flights a user has previously interacted with. It analyzes flight attributes (price, duration, airline, etc.) to find matches.

### Class: `ContentBasedRecommender`

**Location:** `src/content_recommender.py`

#### Constructor

```python
ContentBasedRecommender(n_recommendations: int = 5)
```

**Parameters:**
- `n_recommendations` (int): Number of recommendations to generate per user. Default: 5

#### Methods

##### `fit(flights_df, user_interactions)`

Train the recommender on flight and interaction data.

**Parameters:**
- `flights_df` (pd.DataFrame): Flight data with columns:
  - `flight_id` (str): Unique flight identifier
  - `airline` (str): Airline name
  - `price` (float): Flight price
  - `duration` (int): Flight duration in minutes
  - `departure_time` (str): Departure time
  - `arrival_time` (str): Arrival time
  - `source` (str): Source airport
  - `destination` (str): Destination airport
  
- `user_interactions` (pd.DataFrame): User booking data with columns:
  - `user_id` (str): Unique user identifier
  - `flight_id` (str): Flight identifier
  - `interaction_type` (str): 'view', 'search', 'booking', etc.

**Returns:** Self (for method chaining)

**Example:**
```python
from content_recommender import ContentBasedRecommender

recommender = ContentBasedRecommender(n_recommendations=5)
recommender.fit(flights_df, user_interactions)
```

##### `recommend(user_id: str, n_recommendations: int = None)`

Generate recommendations for a specific user.

**Parameters:**
- `user_id` (str): User identifier
- `n_recommendations` (int, optional): Override default recommendation count

**Returns:** 
- `list[tuple]`: List of (flight_id, score) tuples sorted by score

**Example:**
```python
recommendations = recommender.recommend('U00001')
for flight_id, score in recommendations:
    print(f"Flight {flight_id}: score={score:.3f}")
```

##### `recommend_batch(user_ids: list[str])`

Generate recommendations for multiple users efficiently.

**Parameters:**
- `user_ids` (list[str]): List of user identifiers

**Returns:** 
- `dict[str, list[tuple]]`: Mapping of user_id to recommendation lists

**Example:**
```python
user_batch = ['U00001', 'U00002', 'U00003']
all_recommendations = recommender.recommend_batch(user_batch)
```

##### `explain_recommendation(user_id: str, flight_id: str)`

Explain why a flight was recommended.

**Parameters:**
- `user_id` (str): User identifier
- `flight_id` (str): Flight identifier

**Returns:** 
- `dict`: Explanation with similarity factors and matching attributes

**Example:**
```python
explanation = recommender.explain_recommendation('U00001', 'F00101')
print(f"Similarity: {explanation['similarity_score']}")
print(f"Shared attributes: {explanation['shared_features']}")
```

---

## Collaborative Recommender

### Overview

The collaborative recommender uses matrix factorization to identify latent factors representing user preferences and flight characteristics. It finds users with similar taste and recommends flights they liked.

### Class: `CollaborativeRecommender`

**Location:** `src/collaborative_recommender.py`

#### Constructor

```python
CollaborativeRecommender(n_factors: int = 20, learning_rate: float = 0.01, 
                        reg_param: float = 0.1, n_iterations: int = 100,
                        n_recommendations: int = 5)
```

**Parameters:**
- `n_factors` (int): Latent factor dimensions. Default: 20
- `learning_rate` (float): SGD learning rate. Default: 0.01
- `reg_param` (float): Regularization parameter. Default: 0.1
- `n_iterations` (int): Training iterations. Default: 100
- `n_recommendations` (int): Recommendations per user. Default: 5

#### Methods

##### `fit(user_flight_matrix)`

Train the collaborative model using matrix factorization.

**Parameters:**
- `user_flight_matrix` (sparse matrix): User-flight interaction matrix

**Returns:** Self

**Example:**
```python
from collaborative_recommender import CollaborativeRecommender
from scipy.sparse import csr_matrix

recommender = CollaborativeRecommender(n_factors=20)
recommender.fit(user_flight_matrix)
```

##### `recommend(user_id: str, n_recommendations: int = None)`

Generate collaborative recommendations.

**Parameters:**
- `user_id` (str): User identifier
- `n_recommendations` (int, optional): Override default count

**Returns:** 
- `list[tuple]`: List of (flight_id, score) tuples

**Example:**
```python
recommendations = recommender.recommend('U00001', n_recommendations=10)
```

##### `find_similar_users(user_id: str, n_similar: int = 5)`

Find users with similar taste.

**Parameters:**
- `user_id` (str): User identifier
- `n_similar` (int): Number of similar users to return. Default: 5

**Returns:** 
- `list[tuple]`: List of (similar_user_id, similarity_score)

**Example:**
```python
similar_users = recommender.find_similar_users('U00001', n_similar=10)
```

---

## Graph-Based Recommender

### Overview

The graph-based recommender builds a user similarity network and recommends flights from similar users' bookings. It's highly scalable and captures local neighborhood patterns.

### Class: `UserSimilarityGraph`

**Location:** `src/graph_recommender.py`

#### Constructor

```python
UserSimilarityGraph(k: int = 5, similarity_metric: str = 'cosine',
                   n_recommendations: int = 5)
```

**Parameters:**
- `k` (int): Number of nearest neighbors per user. Default: 5
- `similarity_metric` (str): Distance metric ('cosine', 'euclidean'). Default: 'cosine'
- `n_recommendations` (int): Recommendations per user. Default: 5

#### Methods

##### `fit(flight_features, user_features, user_bookings)`

Build the user similarity graph.

**Parameters:**
- `flight_features` (pd.DataFrame): Flight feature matrix
- `user_features` (pd.DataFrame): User feature matrix
- `user_bookings` (dict): User to booked flights mapping {user_id: [flight_ids]}

**Returns:** Self

**Example:**
```python
from graph_recommender import UserSimilarityGraph

graph = UserSimilarityGraph(k=5)
graph.fit(flight_features, user_features, user_bookings)
```

##### `recommend(user_id: str, n_recommendations: int = None)`

Generate recommendations using graph neighborhoods.

**Parameters:**
- `user_id` (str): User identifier
- `n_recommendations` (int, optional): Override default count

**Returns:** 
- `list[tuple]`: List of (flight_id, score) tuples

**Example:**
```python
recommendations = graph.recommend('U00001')
```

##### `get_neighbors(user_id: str, n_neighbors: int = None)`

Get the k-nearest neighbors for a user.

**Parameters:**
- `user_id` (str): User identifier
- `n_neighbors` (int, optional): Override k

**Returns:** 
- `list[tuple]`: List of (neighbor_id, similarity_score)

**Example:**
```python
neighbors = graph.get_neighbors('U00001')
for neighbor_id, similarity in neighbors:
    print(f"Neighbor {neighbor_id}: similarity={similarity:.3f}")
```

##### `get_graph_stats()`

Get statistics about the built graph.

**Returns:** 
- `dict`: Graph statistics

**Example:**
```python
stats = graph.get_graph_stats()
print(f"Users: {stats['num_users']}")
print(f"Neighbors per user: {stats['neighbors_per_user']}")
```

---

## Recommender Ensemble

### Overview

The ensemble combines predictions from multiple recommenders using weighted voting. It provides robust, well-rounded recommendations.

### Class: `RecommenderEnsemble`

**Location:** `src/ensemble_recommender.py`

#### Constructor

```python
RecommenderEnsemble(recommenders: dict[str, tuple], 
                   ensemble_method: str = 'weighted_average',
                   n_recommendations: int = 5)
```

**Parameters:**
- `recommenders` (dict): Mapping of {name: (recommender, weight)}
- `ensemble_method` (str): 'weighted_average', 'rank_fusion', 'voting'. Default: 'weighted_average'
- `n_recommendations` (int): Final recommendations per user. Default: 5

#### Methods

##### `__init__` with recommenders

```python
from ensemble_recommender import RecommenderEnsemble
from content_recommender import ContentBasedRecommender
from collaborative_recommender import CollaborativeRecommender

# Create individual recommenders
content_rec = ContentBasedRecommender()
collab_rec = CollaborativeRecommender()

# Create ensemble with weights
ensemble = RecommenderEnsemble(
    recommenders={
        'content': (content_rec, 0.3),
        'collaborative': (collab_rec, 0.7)
    },
    ensemble_method='weighted_average'
)
```

##### `fit(data_dict)`

Train all recommenders in the ensemble.

**Parameters:**
- `data_dict` (dict): Dictionary with data for each recommender

**Returns:** Self

**Example:**
```python
data = {
    'flights_df': flights_df,
    'user_interactions': user_interactions,
    'user_flight_matrix': user_flight_matrix
}
ensemble.fit(data)
```

##### `recommend(user_id: str, n_recommendations: int = None)`

Generate ensemble recommendations.

**Parameters:**
- `user_id` (str): User identifier
- `n_recommendations` (int, optional): Override default count

**Returns:** 
- `list[tuple]`: Consolidated (flight_id, score) tuples

**Example:**
```python
recommendations = ensemble.recommend('U00001')
```

##### `get_component_recommendations(user_id: str)`

Get recommendations from each component recommender.

**Parameters:**
- `user_id` (str): User identifier

**Returns:** 
- `dict`: Mapping of recommender name to recommendation lists

**Example:**
```python
components = ensemble.get_component_recommendations('U00001')
for name, recs in components.items():
    print(f"{name}: {recs}")
```

##### `set_weights(weights: dict[str, float])`

Update ensemble weights dynamically.

**Parameters:**
- `weights` (dict): New weight mapping {recommender_name: weight}

**Returns:** Self

**Example:**
```python
ensemble.set_weights({
    'content': 0.2,
    'collaborative': 0.8
})
```

---

## Data Models

### Flight Data Structure

```python
{
    'flight_id': str,          # Unique identifier
    'airline': str,            # Airline name
    'source': str,             # Departure airport
    'destination': str,        # Arrival airport
    'price': float,            # Flight price
    'duration': int,           # Duration in minutes
    'departure_time': str,     # ISO format time
    'arrival_time': str,       # ISO format time
    'date': str,               # ISO format date
    'stops': int,              # Number of stops
    'aircraft': str            # Aircraft type
}
```

### User Interaction Structure

```python
{
    'user_id': str,            # Unique identifier
    'flight_id': str,          # Flight identifier
    'interaction_type': str,   # 'view', 'search', 'booking'
    'timestamp': str,          # ISO format timestamp
    'rating': float,           # 1-5 (optional)
    'booking_value': float     # Ticket price (optional)
}
```

### Recommendation Output Format

```python
[
    ('F00101', 0.92),          # flight_id, relevance_score (0-1)
    ('F00205', 0.87),
    ('F00312', 0.84),
    ('F00401', 0.79),
    ('F00508', 0.73)
]
```

---

## Usage Examples

### Example 1: Simple Content-Based Recommendations

```python
import pandas as pd
from content_recommender import ContentBasedRecommender

# Load data
flights = pd.read_csv('flights.csv')
interactions = pd.read_csv('interactions.csv')

# Create and train recommender
recommender = ContentBasedRecommender(n_recommendations=5)
recommender.fit(flights, interactions)

# Get recommendations
recommendations = recommender.recommend('U00001')
print(recommendations)
# Output: [('F00101', 0.92), ('F00205', 0.87), ...]
```

### Example 2: Collaborative Filtering with Matrix Factorization

```python
from collaborative_recommender import CollaborativeRecommender
from scipy.sparse import csr_matrix

# Create user-flight interaction matrix
user_flight_matrix = csr_matrix((ratings, (user_indices, flight_indices)))

# Create and train
recommender = CollaborativeRecommender(n_factors=20, n_iterations=100)
recommender.fit(user_flight_matrix)

# Find similar users
similar_users = recommender.find_similar_users('U00001', n_similar=5)
print(similar_users)
```

### Example 3: Graph-Based Recommendations

```python
from graph_recommender import UserSimilarityGraph

# Build graph
graph = UserSimilarityGraph(k=5)
graph.fit(flight_features, user_features, user_bookings)

# Get recommendations
recommendations = graph.recommend('U00001')

# Inspect neighborhood
neighbors = graph.get_neighbors('U00001')
print(f"Similar users: {neighbors}")

# Get statistics
stats = graph.get_graph_stats()
print(f"Graph has {stats['num_users']} users")
```

### Example 4: Ensemble Approach

```python
from ensemble_recommender import RecommenderEnsemble
from content_recommender import ContentBasedRecommender
from collaborative_recommender import CollaborativeRecommender
from graph_recommender import UserSimilarityGraph

# Create individual recommenders
content = ContentBasedRecommender()
collab = CollaborativeRecommender()
graph = UserSimilarityGraph()

# Create ensemble
ensemble = RecommenderEnsemble(
    recommenders={
        'content': (content, 0.3),
        'collaborative': (collab, 0.5),
        'graph': (graph, 0.2)
    },
    ensemble_method='weighted_average'
)

# Train all recommenders
ensemble.fit({
    'flights_df': flights,
    'user_interactions': interactions,
    'user_flight_matrix': user_flight_matrix,
    'flight_features': flight_features,
    'user_features': user_features,
    'user_bookings': user_bookings
})

# Get ensemble recommendations
recommendations = ensemble.recommend('U00001')
print(recommendations)

# See individual component scores
components = ensemble.get_component_recommendations('U00001')
for name, recs in components.items():
    print(f"{name}: {recs}")
```

### Example 5: Batch Processing

```python
# Process multiple users efficiently
user_batch = ['U00001', 'U00002', 'U00003', 'U00004', 'U00005']

# Content-based batch
content_recs = recommender.recommend_batch(user_batch)

# Graph-based batch  
graph_recs = graph.get_batch_recommendations(user_batch)

# Process results
for user_id, recommendations in content_recs.items():
    print(f"{user_id}: {recommendations}")
```

---

## Performance Considerations

### Recommender Selection

| Recommender | Memory | Speed | Scalability | Best For |
|-------------|--------|-------|-------------|----------|
| Content-Based | Low | Fast | Excellent | Cold-start, new users |
| Collaborative | Medium | Medium | Good | Dense interactions |
| Graph-Based | Low | Fast | Excellent | Large systems, local patterns |
| Ensemble | High | Slow | Good | High accuracy, robustness |

### Optimization Tips

1. **Content-Based**: Pre-compute feature matrices, vectorize similarity calculations
2. **Collaborative**: Use sparse matrices, tune factor dimensions
3. **Graph-Based**: Cache similarity matrix, use approximate nearest neighbors
4. **Ensemble**: Cache component recommendations, use multiprocessing

---

## Error Handling

### Common Exceptions

```python
try:
    recommendations = recommender.recommend('U00999')
except ValueError as e:
    print(f"User not found: {e}")
except RuntimeError as e:
    print(f"Model not fitted: {e}")
```

### Validation

```python
# Check if recommender is ready
if not recommender.is_fitted():
    raise RuntimeError("Recommender needs to be fitted first")
```

---

## Advanced Features

### Explanation and Interpretability

```python
# Why was this flight recommended?
explanation = recommender.explain_recommendation('U00001', 'F00101')
print(explanation)
# {
#   'similarity_score': 0.92,
#   'matching_attributes': ['price_range', 'airline', 'route'],
#   'shared_features': {'airline': 'Delta', 'duration': '2h-3h'}
# }
```

### Dynamic Weight Adjustment

```python
# Adjust ensemble weights based on feedback
feedback_scores = {'content': 0.2, 'collaborative': 0.8}
ensemble.set_weights(feedback_scores)
```

---

## Troubleshooting

### Issue: Poor Recommendation Quality

**Solutions:**
1. Check data quality (missing values, outliers)
2. Increase training iterations
3. Adjust hyperparameters (learning rate, regularization)
4. Use ensemble method for more robust results

### Issue: Memory Issues

**Solutions:**
1. Use sparse matrices for large interaction data
2. Reduce embedding dimensions
3. Use graph-based recommender (memory efficient)
4. Process in batches

### Issue: Slow Performance

**Solutions:**
1. Use approximate nearest neighbor methods
2. Cache computed similarities
3. Reduce k (neighbors count) in graph recommender
4. Use multiprocessing for batch recommendations

---

## Additional Resources

- [Feature Engineering Guide](./FEATURE_ENGINEERING.md)
- [Evaluation Metrics](./EVALUATION_METRICS.md)
- [Performance Tuning](./PERFORMANCE_TUNING.md)
- [Example Notebooks](../notebooks/)

---

**Last Updated:** 2024
**Version:** 1.0.0
