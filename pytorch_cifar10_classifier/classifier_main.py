# from https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


#############################################################################
# Load the data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 8

train_dataset = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=True, transform=transform
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

test_dataset = torchvision.datasets.CIFAR10(
    root="../data", train=False, download=True, transform=transform
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)


classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


##########################################################################################
# Downsample the dataset
#

train_dataset.data = train_dataset.data[:10000]  # Keep only the first 10000 samples for faster training
train_dataset.targets = train_dataset.targets[:10000]



###########################################################################################
# Data visualization
# 

display_data = False  # Set to True to display images
if display_data:
    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)

    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

    # show images
    imshow(torchvision.utils.make_grid(images))

###########################################################################################
# Define the model
#

class convolutional_model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
model = convolutional_model().to(device)


loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-8)


epoch_count = 20
for epoch in range(epoch_count):  # loop over the dataset multiple times

    running_loss = 0.0
    for batch_index, data in enumerate(train_dataloader, 0):
        # print(f"Batch {i} of {len(train_dataloader)}")
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        n_batch_per_log = 100
        if batch_index % n_batch_per_log == n_batch_per_log - 1:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {batch_index + 1:5d}] loss: {running_loss / n_batch_per_log:.3f}")
            running_loss = 0.0

print(f"Finished Training - {epoch_count} epochs loss: {loss.item():.3f}")
