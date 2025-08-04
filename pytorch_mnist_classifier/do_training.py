# Importing dependencies
import torch
from PIL import Image
from torch import nn,save,load
# from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ImageClassifierModel  # Importing the model from the other file

import os
__dirname__ = os.path.dirname(os.path.abspath(__file__))

######################################################################################
# Load the dataset
#

# Loading Data
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root=os.path.join(__dirname__, "../data"), download=True, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.FashionMNIST(root=os.path.join(__dirname__, "../data"), download=True, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

######################################################################################
# downsample the dataset for faster training (good for dev training)
#

# limit the dataset to 1000 samples for faster training
# train_dataset.data = train_dataset.data[:10000]

##################################################################################
# Create an instance of the image classifier model
classifier_model = ImageClassifierModel()

ImageClassifierModel.do_train(classifier_model, train_loader, epochs=50, log_enabled=True)

##################################################################################
# Save the trained model to a file
torch.save(classifier_model.state_dict(), os.path.join(__dirname__, './data/model_classifier.pth'))

##################################################################################
# Evaluate the model on the test dataset
#

test_accuracy = ImageClassifierModel.do_eval(classifier_model, test_loader)
print(f'Accuracy of the model on the test images: {test_accuracy:.2f}%')

train_accuracy = ImageClassifierModel.do_eval(classifier_model, train_loader)
print(f'Accuracy of the model on the train images: {train_accuracy:.2f}%')