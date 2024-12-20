import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

X = data.iloc[:, :-1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = range(2, 10)
inertia = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

optimal_k = k_values[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)

affinity = AffinityPropagation(random_state=42)
affinity.fit(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(k_values, inertia, marker='o')
axes[0].set_title("K-Means Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")

axes[1].plot(k_values, silhouette_scores, marker='o')
axes[1].set_title("K-Means Silhouette Scores")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.show()

print("K-Means Optimal Number of Clusters:", optimal_k)
print("Affinity Propagation Number of Clusters:", len(np.unique(affinity.labels_)))
