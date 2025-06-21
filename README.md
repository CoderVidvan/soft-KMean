# Soft KMeans / Fuzzy C-Means Implementation

```python
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
```

## Generate Sample Data

```python
# Random seed for reproducibility
np.random.seed(42)

# Generate some sample 2D data (3 clusters)
data1 = np.random.normal(loc=[2, 2], scale=0.5, size=(100, 2))
data2 = np.random.normal(loc=[6, 6], scale=0.5, size=(100, 2))
data3 = np.random.normal(loc=[10, 2], scale=0.5, size=(100, 2))

data = np.vstack((data1, data2, data3)).T  # Shape (2, 300)
```

## Apply Fuzzy C-Means Clustering

```python
# Parameters
n_clusters = 3
m = 2.0  # Fuzziness
error = 0.005
maxiter = 1000

# Run FCM
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None)
```

## Visualize Final Cluster Assignments

```python
# Hard clustering result
cluster_labels = np.argmax(u, axis=0)

# Plot data and centers
fig, ax = plt.subplots()
colors = ['r', 'g', 'b']

for i in range(n_clusters):
    ax.scatter(data[0, cluster_labels == i], data[1, cluster_labels == i], 
               label=f'Cluster {i+1}', alpha=0.6)
    ax.plot(cntr[i, 0], cntr[i, 1], 'kx', markersize=15)

ax.set_title("Fuzzy C-Means Clustering")
ax.legend()
plt.show()
```

## Compute Cluster Radii (Weighted by Membership)

```python
# Compute fuzzy radius per cluster
radii = []
for i in range(n_clusters):
    weights = u[i]
    weighted_squared_dist = weights * d[i]**2
    radius = np.sqrt(np.sum(weighted_squared_dist) / np.sum(weights))
    radii.append(radius)
```

## Draw Circles for Each Cluster

```python
fig, ax = plt.subplots()

# Plot points
for i in range(n_clusters):
    ax.scatter(data[0, cluster_labels == i], data[1, cluster_labels == i], alpha=0.5)
    ax.plot(cntr[i, 0], cntr[i, 1], 'ko')

    # Draw fuzzy boundary as circle
    circle = Circle((cntr[i, 0], cntr[i, 1]), radii[i], 
                    fill=False, linestyle='--', edgecolor='black')
    ax.add_patch(circle)

ax.set_title("Clusters with Fuzzy Boundaries")
ax.set_aspect('equal')
plt.show()
```

## Fuzzy Partition Coefficient (FPC)

```python
print(f"Fuzzy Partition Coefficient (FPC): {fpc:.4f}")
```

---

### ✅ Summary

- **Soft KMeans** is implemented using `scikit-fuzzy`'s `cmeans`.
- We calculate the **fuzzy radius** for each cluster based on weighted distances.
- Circles represent the **fuzzy boundary** around each cluster center.
- `fpc` is a metric to assess how well-defined the clusters are (closer to 1 is better).

