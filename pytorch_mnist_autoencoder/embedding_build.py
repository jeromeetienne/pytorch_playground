# from https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from autoencoder_model import Autoencoder_model_linear, Autoencoder_model_conv2d  # Importing the model from the other file
import numpy as np
import time
import umap
import matplotlib.pyplot as plt
import termcolor

import os
__dirname__ = os.path.dirname(os.path.abspath(__file__))

######################################################################################
# Load the MNIST dataset
# and create a DataLoader for training the autoencoder
# 

tensor_transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root=os.path.join(__dirname__, "../data"), train=True, download=True, transform=tensor_transform)
test_dataset = datasets.MNIST(root=os.path.join(__dirname__, "../data"), train=False, download=True, transform=tensor_transform)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

######################################################################################
# downsample the dataset for faster training (good for dev training)
#

# limit the dataset to 1000 samples for faster training
# train_dataset.data = train_dataset.data[:1000]

######################################################################################
# define the autoencoder model, and load the trained model
#

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# device = 'cpu'

# Instantiate the model and move it to the device
# model = Autoencoder_model_linear()
model = Autoencoder_model_conv2d()
model.to(device)

print(f"Using {termcolor.colored(device, 'cyan')} device for building embedding {termcolor.colored(model.model_name, 'cyan')} model...")

# Load the model from a file
model_filename = os.path.join(__dirname__, f'./data/{model.model_name}.pth')
model.load_state_dict(torch.load(model_filename))
print(f"Model loaded {termcolor.colored(model.model_name, 'cyan')}")

######################################################################################
# Visualize the reconstructed images
#

image_embeddings = None
image_labels = None

for loaded_images, loaded_labels in test_dataloader:
    original_images = loaded_images.to(device)
    embeddings = model.encoder(original_images)
    # initialize the image_embeddings and image_labels if they are None
    if image_embeddings is None:
        image_embeddings = np.empty((0, embeddings.size(1)), dtype=float)
    if image_labels is None:
        image_labels = np.empty((0,), dtype=int)
    image_labels = np.append(image_labels, loaded_labels.cpu().detach().numpy(), axis=0)
    image_embeddings = np.append(image_embeddings, embeddings.cpu().detach().numpy().reshape(-1, embeddings.size(1)), axis=0)

print("Embeddings shape:", image_embeddings.shape)
print("Labels shape:", image_labels.shape)

#######################################################################################
# write the embeddings to a file
#

embeddings_filename = os.path.join(__dirname__, f'./data/{model.model_name}_embeddings.npz')
np.savez(embeddings_filename, image_embeddings)

print(f"Embeddings saved to {embeddings_filename}")

######################################################################################
# Build the UMAP embedding from the embeddings
#

time_start = time.time()
print("Fitting UMAP...")
umap_fit = umap.UMAP()
# umap_fit = umap.UMAP(random_state=123)
umap_points_fitted = umap_fit.fit_transform(image_embeddings)
time_elapsed = time.time() - time_start
print(f"UMAP fit_transform took {time_elapsed:.2f} seconds")

######################################################################################
# save fitted points in file

fitted_points_filename = os.path.join(__dirname__, f'./data/{model.model_name}_fitted_points.npz')
np.savez(fitted_points_filename, umap_points_fitted)

######################################################################################
# Compute the colors for the UMAP points based on labels
#
umap_colors = np.empty((len(umap_points_fitted), 4))
umap_cmap = plt.get_cmap('viridis')
for i in range(len(umap_points_fitted)):
    umap_colors[i] = umap_cmap(image_labels[i] / 10)  # Normalize label for colormap


#######################################################################################
# Visualize the UMAP embedding

plt.figure(figsize=(10, 10))
plt.scatter(umap_points_fitted[:,0], umap_points_fitted[:,1], c=umap_colors, s=1)

for label in range(len(np.unique(image_labels))):
    print(f"Computing center for label {label}")
    label_indices = np.where(image_labels == label)[0]
    label_points = umap_points_fitted[label_indices]
    label_center = np.mean(label_points, axis=0)
    plt.text(label_center[0], label_center[1], str(label), fontsize=30, ha='center', va='center', color='red', fontweight='bold', alpha=0.5)

plt.title(f"UMAP Embedding of {model.model_name} Embeddings of MNIST")

# Save the image in a file
image_filename = os.path.join(__dirname__, f'./data/{model.model_name}_umap.png')
plt.savefig(image_filename)

plt.show(block=True)

