# from https://www.perplexity.ai/search/how-to-eval-a-mnist-classifier-XS37oDlZRL6_x9aPAksQGQ

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import ImageClassifierModel  # Importing the model from the other file

import os
__dirname__ = os.path.dirname(os.path.abspath(__file__))

######################################################################################
# Load the MNIST dataset
# and create a DataLoader for training the autoencoder
# 

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root=os.path.join(__dirname__, "../data"), train=True, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(root=os.path.join(__dirname__, "../data"), train=False, download=True, transform=transform)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

######################################################################################
# define the autoencoder model, and load the trained model
#

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Instantiate the model and move it to the device
model = ImageClassifierModel()
model.to(device)

# Load the model from a file
model_filename = os.path.join(__dirname__, './model_classifier.pth')
model.load_state_dict(torch.load(model_filename))

######################################################################################
# Visualize the reconstructed images
#

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_dataloader:  # test_loader is the DataLoader for the test dataset
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)  # Forward pass

        _, predicted = torch.max(outputs, 1)  # Get predicted class from output probabilities
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')


