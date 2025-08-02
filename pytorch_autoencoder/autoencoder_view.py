# from https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/

import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from autoencoder_model import Autoencoder_model  # Importing the model from the other file

import os
__dirname__ = os.path.dirname(os.path.abspath(__file__))

tensor_transform = transforms.ToTensor()
dataset = datasets.MNIST(root=os.path.join(__dirname__, "./data"), train=True, download=True, transform=tensor_transform)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move it to the device
model = Autoencoder_model()
model.to(device)

# Load the model from a file
model_filename = os.path.join(__dirname__, './data/autoencoder_model.pth')
model.load_state_dict(torch.load(model_filename))


for images, _ in data_loader:
    original_images = images.view(-1, 28 * 28).to(device)
    reconstructed_images = model(original_images)

    last_images = original_images.view(-1, 28, 28).cpu().detach().numpy()
    last_reconstructed = reconstructed_images.view(-1, 28, 28).cpu().detach().numpy()

    batch_size = original_images.size(0)

    plt.figure(figsize=(10, 5))
    for i in range(batch_size):
        # Display original images
        plt.subplot(2, batch_size, i + 1)
        plt.imshow(last_images[i], cmap='gray')
        plt.axis('off')

        # Display reconstructed images
        plt.subplot(2, batch_size, i + batch_size + 1)
        plt.imshow(last_reconstructed[i], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

