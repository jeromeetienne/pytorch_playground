# from https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/

import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from autoencoder_model import Autoencoder_model  # Importing the model from the other file

import os
__dirname__ = os.path.dirname(os.path.abspath(__file__))

######################################################################################
# Load the MNIST dataset
# and create a DataLoader for training the autoencoder
# 

tensor_transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root=os.path.join(__dirname__, "../data"), train=True, download=True, transform=tensor_transform)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(root=os.path.join(__dirname__, "../data"), train=False, download=True, transform=tensor_transform)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

######################################################################################
# downsample the dataset for faster training (good for dev training)
#

# limit the dataset to 1000 samples for faster training
# train_dataset.data = train_dataset.data[:1000]

######################################################################################
# Define the autoencoder model, loss function, optimizer
#
model = Autoencoder_model()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 30
outputs = []
losses = []

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = 'cpu'
print(f"Using {device} device")
model.to(device)

######################################################################################
# Train the autoencoder
#

for epoch in range(epochs):
    for batch_index, data_loaded in enumerate(train_dataloader, 0):
        images, labels = data_loaded
        images = images.view(-1, 28 * 28).to(device)

        reconstructed_images = model(images)
        loss = loss_function(reconstructed_images, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # print(f"Epoch {epoch+1}/{epochs}, Batch {batch_index+1}/{len(train_dataloader)}, Loss: {loss.item():.6f}")

    outputs.append((epoch, images, reconstructed_images))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

######################################################################################
# Save the trained model
#

# Save the model to a file
model_filename = os.path.join(__dirname__, './data/autoencoder_model.pth')
torch.save(model.state_dict(), model_filename)

######################################################################################
# Plot the training loss
#

plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()
