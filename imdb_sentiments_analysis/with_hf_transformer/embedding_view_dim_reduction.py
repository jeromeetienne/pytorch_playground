import numpy as np

import time
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from embedding_model_name import ModelNameEnum
import os
import argparse


########################################################################
# command line argument parsing

cmdline_argparser = argparse.ArgumentParser(
    description="IMDB Sentiment Analysis - Dimensionality Reduction View",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
cmdline_argparser.add_argument(
    "--downsample_fraction",
    type=float,
    default=1.0,
    help="Downsample fraction - 1.0 means no downsampling = 0.1 means keep only 10%% of the data",
)
cmdline_argparser.add_argument(
    "--model_name",
    type=str,
    choices=[e.value for e in ModelNameEnum],
    default=f"{ModelNameEnum.ALL_MINILM_L6_V2}",
    help="Model name for text embedding",
)
cmdline_argparser.add_argument(
    "--fit_type",
    type=str,
    choices=["t-SNE", "UMAP"],
    default="UMAP",
    help="Dimensionality reduction method to use: t-SNE or UMAP",
)
cmdline_argparser.add_argument(
    "--ui_show_blocking",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show matplotlib UI in blocking mode (default: True)",
)
cmdline_args = cmdline_argparser.parse_args()


########################################################################
# setup __dirname__

__dirname__ = os.path.dirname(os.path.abspath(__file__))

#################################################################################
# define model name

model_name = cmdline_args.model_name

########################################################################
# Load the embeddings
#

embedding_data = np.load(
    os.path.join(__dirname__, f"./output/imdb_embeddings_{model_name}.npz")
)
embedding_vectors = embedding_data["embeddings"]
label_vectors = embedding_data["labels"]

########################################################################
# Downsample the dataset
#

downsample_fraction = cmdline_args.downsample_fraction
if downsample_fraction < 1.0:
    downsample_count = int(
        len(embedding_vectors) * downsample_fraction
    )  # Number of samples to keep
    embedding_vectors = embedding_vectors[:downsample_count]
    label_vectors = label_vectors[:downsample_count]
    print(
        f"Downsampled dataset factor: {downsample_fraction} current size: {embedding_vectors.shape}"
    )
else:
    print("No downsampling")


######################################################################
# Visualize the embeddings
#

fit_type = cmdline_args.fit_type
print(f"Fitting {fit_type}...")
fitting_time_start = time.time()
if fit_type == "t-SNE":
    tsne_fit = TSNE(n_components=2, perplexity=10)
    # Fit the model and transform the data
    points_fitted = tsne_fit.fit_transform(embedding_vectors)
elif fit_type == "UMAP":
    umap_fit = umap.UMAP()
    # umap_fit = umap.UMAP(random_state=123)
    points_fitted = umap_fit.fit_transform(embedding_vectors)
else:
    print("Unknown fit type")

fitting_time_elapsed = time.time() - fitting_time_start
print(f"{fit_type} fit_transform took {fitting_time_elapsed:.2f} seconds")

######################################################################################
# Compute the colors for the UMAP points based on labels
#
point_fitted_colors = np.empty((len(points_fitted), 4))
point_fitted_cmap = plt.get_cmap("viridis")
for i in range(len(points_fitted)):
    point_fitted_colors[i] = point_fitted_cmap(
        0.25 + label_vectors[i] / 2
    )  # Normalize label for colormap


#######################################################################################
# Visualize the UMAP embedding

plt.figure(figsize=(10, 10))
plt.scatter(points_fitted[:, 0], points_fitted[:, 1], c=point_fitted_colors, s=20)

for label in range(len(np.unique(label_vectors))):
    print(f"Computing center for label {label}")
    label_indices = np.where(label_vectors == label)[0]
    label_points = points_fitted[label_indices]
    label_center = np.mean(label_points, axis=0)
    plt.text(
        label_center[0],
        label_center[1],
        str(label),
        fontsize=30,
        ha="center",
        va="center",
        color="red",
        fontweight="bold",
        alpha=0.5,
    )

plt.title(f"UMAP Embeddings of IMDB Reviews")

# Save the image in a file
image_filename = os.path.join(__dirname__, f"./output/embedding_umap_{model_name}.png")
plt.savefig(image_filename)

plt.show(block=cmdline_args.ui_show_blocking)
