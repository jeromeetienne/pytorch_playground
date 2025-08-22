# from sentence_transformers import SentenceTransformer
import sentence_transformers
import pandas as pd
import numpy as np

########################################################################
# setup __dirname__

import os
__dirname__ = os.path.dirname(os.path.abspath(__file__))


########################################################################
# Read the data
#

dataset_filename = os.path.join(
    __dirname__, "../input/imdb-dataset-sentiment-analysis-in-csv-format/Train.csv"
)
train_dataset = pd.read_csv(dataset_filename)

########################################################################
# Downsample the dataset
#

downsample_enabled = False
if downsample_enabled:
    downsample_fraction = 0.1
    train_dataset = train_dataset.sample(frac=downsample_fraction, random_state=42)

#################################################################################
# Load the embedding model

# Load a pre-trained model (e.g. "all-MiniLM-L6-v2")
text_embedding_model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

embedding_length = text_embedding_model.get_sentence_embedding_dimension()
print(f"Embedding length: {embedding_length}")

########################################################################
# Compute embeddings
#

# Compute embeddings for the text column in the dataset
embedding_vectors = text_embedding_model.encode(train_dataset["text"].to_numpy(), show_progress_bar=True)

########################################################################
# Create a dataframe from the embeddings and save it
#

# Convert embedding and label into numpy array
embedding_ndarray = embedding_vectors
label_ndarray = train_dataset["label"].to_numpy()

# Save the numpy arrays
np.savez_compressed(os.path.join(__dirname__, "./output/imdb_embeddings.npz"), embeddings=embedding_ndarray, labels=label_ndarray)