# from https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# import model as ImageClassifierModel
from model import ImageClassifierModel  # Importing the model from the other file


import os
__dirname__ = os.path.dirname(os.path.abspath(__file__))

######################################################################################
# Load the MNIST dataset
# and create a DataLoader for training the autoencoder
# 

tensor_transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root=os.path.join(__dirname__, "../data"), train=True, download=True, transform=tensor_transform)
test_dataset = datasets.MNIST(root=os.path.join(__dirname__, "../data"), train=False, download=True, transform=tensor_transform)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

######################################################################################
# define the autoencoder model, and load the trained model
#

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = 'cpu'

# Instantiate the model and move it to the device
model = ImageClassifierModel()
model.to(device)

# Load the model from a file
model_filename = os.path.join(__dirname__, './model_classifier.pth')
model.load_state_dict(torch.load(model_filename))

######################################################################################
# Visualize the reconstructed images
#

for images, labels in test_dataloader:
    original_images = images.to(device)
    outputs = model(original_images)

    predicted_label = torch.argmax(outputs)

    print('Predicted label:', predicted_label.item(), labels.item())
    # last_images = original_images.view(-1, 28, 28).cpu().detach().numpy()
    # last_reconstructed = reconstructed_images.view(-1, 28, 28).cpu().detach().numpy()

    # batch_size = original_images.size(0)

    # plt.figure(figsize=(10, 5))
    # for i in range(batch_size):
    #     # Display original images
    #     plt.subplot(2, batch_size, i + 1)
    #     plt.imshow(last_images[i], cmap='gray')
    #     plt.axis('off')

    #     # Display reconstructed images
    #     plt.subplot(2, batch_size, i + batch_size + 1)
    #     plt.imshow(last_reconstructed[i], cmap='gray')
    #     plt.axis('off')

    # plt.tight_layout()
    # plt.show()

