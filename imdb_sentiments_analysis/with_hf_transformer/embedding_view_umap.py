import numpy as np

import time
import umap
import matplotlib.pyplot as plt


########################################################################
# setup __dirname__

import os

__dirname__ = os.path.dirname(os.path.abspath(__file__))

########################################################################
# Load the embeddings
#

embedding_data = np.load(os.path.join(__dirname__, "./output/imdb_embeddings.npz"))
embedding_vectors = embedding_data["embeddings"]
label_vectors = embedding_data["labels"]

######################################################################
# downsample the embeddings
#

downsample_count = 1000  # Number of samples to keep
embedding_vectors = embedding_vectors[:downsample_count]
label_vectors = label_vectors[:downsample_count]

######################################################################
# Visualize the embeddings
#

time_start = time.time()
print("Fitting UMAP...")
umap_fit = umap.UMAP()
# umap_fit = umap.UMAP(random_state=123)
umap_points_fitted = umap_fit.fit_transform(embedding_vectors)
time_elapsed = time.time() - time_start
print(f"UMAP fit_transform took {time_elapsed:.2f} seconds")


######################################################################################
# Compute the colors for the UMAP points based on labels
#
umap_colors = np.empty((len(umap_points_fitted), 4))
umap_cmap = plt.get_cmap("viridis")
for i in range(len(umap_points_fitted)):
    umap_colors[i] = umap_cmap(label_vectors[i] / 2)  # Normalize label for colormap


#######################################################################################
# Visualize the UMAP embedding

plt.figure(figsize=(10, 10))
plt.scatter(umap_points_fitted[:, 0], umap_points_fitted[:, 1], c=umap_colors, s=20)

for label in range(len(np.unique(label_vectors))):
    print(f"Computing center for label {label}")
    label_indices = np.where(label_vectors == label)[0]
    label_points = umap_points_fitted[label_indices]
    label_center = np.mean(label_points, axis=0)
    plt.text(label_center[0], label_center[1], str(label), fontsize=30, ha='center', va='center', color='red', fontweight='bold', alpha=0.5)

plt.title(f"UMAP Embeddings of IMDB Reviews")

# Save the image in a file
image_filename = os.path.join(__dirname__, f"./output/embedding_umap.png")
plt.savefig(image_filename)

plt.show(block=True)
