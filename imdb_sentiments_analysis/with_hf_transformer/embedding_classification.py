import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

########################################################################
# setup __dirname__

import os

__dirname__ = os.path.dirname(os.path.abspath(__file__))

########################################################################
# Load the embeddings
#

embedding_data = np.load(os.path.join(__dirname__, "./output/imdb_embeddings.npz"))
embedding_vectors = embedding_data["embeddings"]
label_vectors = embedding_data["labels"]

########################################################################
# Downsample the embeddings

downsample_enabled = True
if downsample_enabled:
    downsample_count = 10000  # Number of samples to keep
    embedding_vectors = embedding_vectors[:downsample_count]
    label_vectors = label_vectors[:downsample_count]

########################################################################
# Example params

input_dim = 384 # Dimension of the embeddings
num_classes = 2  # Number of classes to classify

batch_size = 4
num_epochs = 120  # Number of training epochs

# Dummy dataset (replace with your actual dataset)
# X = torch.randn(100, input_dim)  # 100 samples
# y = torch.randint(0, num_classes, (100,))  # Class labels for samples

X = torch.tensor(embedding_vectors, dtype=torch.float32)
y = torch.tensor(label_vectors, dtype=torch.long)


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device for training model...")


# Simple model: linear layer
model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),  
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, num_classes),
)

model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0001)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    # Mini-batch loop
    for i in range(0, len(X), batch_size):
        inputs = X[i:i+batch_size].to(device)
        labels = y[i:i+batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / (len(X) / batch_size)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
