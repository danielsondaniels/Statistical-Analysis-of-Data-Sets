from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

data = load_iris()
X = data['data']
y = data['target']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

affinity = AffinityPropagation(random_state=42)
affinity_labels = affinity.fit_predict(X_scaled)
affinity_silhouette = silhouette_score(X_scaled, affinity_labels)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', label='K-Means')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.title("K-Means Clustering")
plt.legend()
plt.show()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=affinity_labels, cmap='coolwarm')
plt.title("Affinity Propagation Clustering")
plt.show()

print("Silhouette Score - K-Means:", kmeans_silhouette)
print("Silhouette Score - Affinity Propagation:", affinity_silhouette)
