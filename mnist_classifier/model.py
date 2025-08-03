# Importing dependencies
from torch import nn


# Define the image classifier model
class ImageClassifierModel(nn.Module):
    def __init__(self):
        super(ImageClassifierModel, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fully_connected_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.fully_connected_layers(x)
        return x
