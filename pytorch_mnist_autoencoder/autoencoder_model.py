# from https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/

import torch
from torch import nn

class Autoencoder_model_linear(torch.nn.Module):
    def __init__(self):
        super(Autoencoder_model_linear, self).__init__()
        self.model_name = "Autoencoder_model_linear"
        self.encoder = nn.Sequential(
            nn.Flatten(),  # Flatten the input
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 9),
        )
        self.decoder = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        # x = x.view(-1, 28 * 28)
        # print(f"Input shape: {x.shape}")
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Define the convolutional autoencoder architecture
class Autoencoder_model_conv2d(nn.Module):
    def __init__(self):
        super(Autoencoder_model_conv2d, self).__init__()
        self.model_name = "Autoencoder_model_conv2d"
        
        # Encoder: Conv layers + downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7)                       # 7x7 -> 1x1 feature map (bottleneck)
        )
        
        # Decoder: ConvTranspose layers + upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),             # 1x1 -> 7x7
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 7x7 -> 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # 14x14 -> 28x28
            nn.Sigmoid()  # Output pixels between 0 and 1
        )
        
    def forward(self, x):
        # x = x.view(-1, 28 * 28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
