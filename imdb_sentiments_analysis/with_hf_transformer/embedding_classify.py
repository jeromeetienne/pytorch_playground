import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sentence_transformers
import os
import argparse
from embedding_model_name import ModelNameEnum


########################################################################
# command line argument parsing

cmdline_argparser = argparse.ArgumentParser(
    description="IMDB Sentiment Analysis - Embedding Classification",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
cmdline_argparser.add_argument(
    "--downsample_fraction",
    type=float,
    default=1.0,
    help="Downsample fraction - 1.0 means no downsampling = 0.1 means keep only 10%% of the data",
)
cmdline_argparser.add_argument(
    "--model_name",
    type=str,
    choices=[e.value for e in ModelNameEnum],
    default=f"{ModelNameEnum.ALL_MINILM_L6_V2}",
    help="Model name for text embedding",
)
cmdline_argparser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="Batch size for training",
)
cmdline_argparser.add_argument(
    "--num_epochs",
    type=int,
    default=120,
    help="Number of training epochs",
)
cmdline_argparser.add_argument(
    "--learning_rate",
    type=float,
    default=0.003,
    help="Learning rate for the optimizer",
)
cmdline_argparser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0001,
    help="Weight decay (L2 regularization) for the optimizer",
)
cmdline_args = cmdline_argparser.parse_args()

########################################################################
# setup __dirname__

__dirname__ = os.path.dirname(os.path.abspath(__file__))

#################################################################################
# define model name

model_name = cmdline_args.model_name

#################################################################################
# Load the embedding model


# Load a pre-trained model (e.g. "all-MiniLM-L6-v2")
text_embedding_model = sentence_transformers.SentenceTransformer(model_name)
embedding_length = text_embedding_model.get_sentence_embedding_dimension()
print(f"model: {model_name} Embedding length: {embedding_length}")

########################################################################
# Load the embeddings
#

embedding_data = np.load(
    os.path.join(__dirname__, f"./output/imdb_embeddings_{model_name}.npz")
)
embedding_vectors = embedding_data["embeddings"]
label_vectors = embedding_data["labels"]

########################################################################
# Downsample the dataset
#

downsample_fraction = cmdline_args.downsample_fraction
if downsample_fraction < 1.0:
    downsample_count = int(
        len(embedding_vectors) * downsample_fraction
    )  # Number of samples to keep
    embedding_vectors = embedding_vectors[:downsample_count]
    label_vectors = label_vectors[:downsample_count]
    print(
        f"Downsampled dataset factor: {downsample_fraction} current size: {embedding_vectors.shape}"
    )
else:
    print("No downsampling")


########################################################################
# Example params

input_dim = embedding_length  # Dimension of the embeddings
batch_size = cmdline_args.batch_size  # Batch size for training
num_epochs = cmdline_args.num_epochs  # Number of training epochs
num_classes = 2  # Number of classes to classify

# Prepare the dataset from embeddings and labels
tensors_x = torch.tensor(embedding_vectors, dtype=torch.float32)
tensors_y = torch.tensor(label_vectors, dtype=torch.long)


device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
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
learning_rate = cmdline_args.learning_rate
weight_decay = cmdline_args.weight_decay
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    # Mini-batch loop
    for i in range(0, len(tensors_x), batch_size):
        inputs = tensors_x[i : i + batch_size].to(device)
        labels = tensors_y[i : i + batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / (len(tensors_x) / batch_size)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
