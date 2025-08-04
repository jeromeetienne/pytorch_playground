# from https://www.perplexity.ai/search/how-to-eval-a-mnist-classifier-XS37oDlZRL6_x9aPAksQGQ

import torch
from torchvision import datasets, transforms
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

# Instantiate the model and move it to the device
model = ImageClassifierModel()

# Load the model from a file
model_filename = os.path.join(__dirname__, './data/model_classifier.pth')
model.load_state_dict(torch.load(model_filename))

######################################################################################
# Visualize the reconstructed images
#

test_accuracy = ImageClassifierModel.do_eval(model, test_dataloader)
print(f'Accuracy of the model on the test images: {test_accuracy:.2f}%')

train_accuracy = ImageClassifierModel.do_eval(model, train_dataloader)
print(f'Accuracy of the model on the train images: {train_accuracy:.2f}%')



