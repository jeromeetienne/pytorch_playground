# Importing dependencies
import torch
from PIL import Image
from torch import nn,save,load
# from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import model as ImageClassifierModel
from model import ImageClassifierModel  # Importing the model from the other file

import os
__dirname__ = os.path.dirname(os.path.abspath(__file__))

######################################################################################
# Load the dataset
#

# Loading Data
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root=os.path.join(__dirname__, "../data"), download=True, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(root=os.path.join(__dirname__, "../data"), download=True, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

######################################################################################
# downsample the dataset for faster training (good for dev training)
#

# limit the dataset to 1000 samples for faster training
train_dataset.data = train_dataset.data[:30000]

# Create an instance of the image classifier model
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
classifier_model = ImageClassifierModel().to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

epoch_count = 30  # Number of epochs to train

# Train the model
for epoch in range(epoch_count):  # Train for 10 epochs
    for images, labels in train_loader:
        # Move images and labels to the device
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()               # Reset gradients
        outputs = classifier_model(images)  # Forward pass
        loss = loss_fn(outputs, labels)     # Compute loss
        loss.backward()                     # Backward pass
        optimizer.step()                    # Update weights

    print(f"Epoch:{epoch} loss is {(loss.item()*100):.4f}%")

# Save the trained model
torch.save(classifier_model.state_dict(), 'model_classifier.pth')

#################################################################################################################
# Inference code
#################################################################################################################

# # Load the saved model
# with open('model_classifier.pth', 'rb') as f: 
#      classifier_model.load_state_dict(load(f))  

# # Perform inference on an image
# img = Image.open('image.jpg')
# img_transform = transforms.Compose([transforms.ToTensor()])
# img_tensor = img_transform(img).unsqueeze(0).to(device)
# output = classifier_model(img_tensor)
# predicted_label = torch.argmax(output)
# print(f"Predicted label: {predicted_label}")