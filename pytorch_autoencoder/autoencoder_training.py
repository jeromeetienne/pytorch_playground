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
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)


model = Autoencoder_model()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

epochs = 30
outputs = []
losses = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


for epoch in range(epochs):
    for images, _ in data_loader:
        images = images.view(-1, 28 * 28).to(device)
        
        reconstructed = model(images)
        loss = loss_function(reconstructed, images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

    outputs.append((epoch, images, reconstructed))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

#############################################

# Save the model to a file
model_filename = os.path.join(__dirname__, './data/autoencoder_model.pth')
torch.save(model.state_dict(), model_filename)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()
