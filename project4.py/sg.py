import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = {
    'Number of Sales': [5, 7, 6, 4, 8, 20, 22, 25, 18, 21, 40, 42, 45, 38, 50],
    'Profit': [2, 3, 2.5, 1.8, 3.2, 10, 12, 11, 9, 13, 25, 28, 30, 22, 32]
}
df = pd.DataFrame(data)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(df)
df['Cluster'] = kmeans.labels_
plt.figure(figsize=(8, 6))
plt.scatter(df['Number of Sales'], df['Profit'], c=df['Cluster'], cmap='cool', s=100)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='green', s=200, marker='s', label='Centroids')
plt.title('K-Means Clustering: Number of Sales vs Profit')
plt.xlabel('Number of Sales')
plt.ylabel('Profit')
plt.legend()
plt.grid(True)
plt.show()