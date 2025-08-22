# from sentence_transformers import SentenceTransformer
import sentence_transformers
import pandas as pd
import numpy as np
from embedding_model_name import ModelNameEnum
import os
import argparse

########################################################################
# command line argument parsing

cmdline_argparser = argparse.ArgumentParser(
    description="IMDB Sentiment Analysis - Embedding Build",
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
cmdline_args = cmdline_argparser.parse_args()

########################################################################
# setup __dirname__

__dirname__ = os.path.dirname(os.path.abspath(__file__))

########################################################################
# Read the data
#

dataset_filename = os.path.join(
    __dirname__, "../input/imdb-dataset-sentiment-analysis-in-csv-format/Train.csv"
)
train_dataset_df = pd.read_csv(dataset_filename)

########################################################################
# Downsample the dataset
#

downsample_fraction = cmdline_args.downsample_fraction
if downsample_fraction < 1.0:
    train_dataset_df = train_dataset_df.sample(
        frac=downsample_fraction, random_state=42
    )
    print(
        f"Downsampled dataset factor: {downsample_fraction} current size: {train_dataset_df.shape}"
    )
else:
    print("No downsampling")

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
# Compute embeddings
#

# Compute embeddings for the text column in the dataset
embedding_vectors = text_embedding_model.encode(
    train_dataset_df["text"].to_numpy(), show_progress_bar=True
)

########################################################################
# Create a dataframe from the embeddings and save it
#

# Convert embedding and label into numpy array
embedding_ndarray = embedding_vectors
label_ndarray = train_dataset_df["label"].to_numpy()

# Save the numpy arrays
embedding_file_name = f"./output/imdb_embeddings_{model_name}.npz"
np.savez_compressed(
    os.path.join(__dirname__, embedding_file_name),
    embeddings=embedding_ndarray,
    labels=label_ndarray,
)
print(f"Saved embeddings to {embedding_file_name}")
