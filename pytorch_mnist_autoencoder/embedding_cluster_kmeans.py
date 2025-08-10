import os
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

n_classes = 10

# Load the embeddings
__dirname__ = os.path.dirname(os.path.abspath(__file__))
embeddings_filename = os.path.join(__dirname__, f'./data/Autoencoder_model_conv2d_fitted_points.npz')
file_data = np.load(embeddings_filename)
X = file_data['arr_0']


# Create a KMeans model for 10 clusters
kmeans = KMeans(n_clusters=n_classes, init='k-means++', n_init=5, max_iter=300)

# Fit the model to the data
kmeans.fit(X)

# Predict the cluster labels for each point
y_kmeans = kmeans.predict(X)

# Plot the data points colored by cluster label
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=5, cmap='viridis')

# Plot the cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering Example')
plt.show()
