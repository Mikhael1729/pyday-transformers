import torch

from data import Data, Datasets, DatasetSplit

torch.manual_seed(1337) # For reproducibility

def main():
  """
  1. Data:

  In this section, we show the raw and processed data that
  the model will use for training.

  Review each line and its output to get a grasp of the
  basic API for data handling in the project.
  """
  data = Data("./input.txt")
  
  # Let's see a chunk of the raw data
  print(data.text[:175])

  # This is a character-level language model, so with this we can visualize the encoding map
  print(data.stoi)

  # Vocabulary size
  print(len(data.unique_chars))

  # Encode the data acording to stoi and store it in a tensor
  datasets = Datasets(torch.tensor(data.encode(data.text), dtype=torch.long))

  # Let's see a chunk of the processed training data
  Xb, Yb = datasets.get_batch(DatasetSplit.train)
  print(Xb)
  print(Yb)

  # Let's use the decode function to see it better:
  print(data.decode(Xb[1].tolist()))
  print(data.decode(Yb[1].tolist()))

  # <-- Play with BLOCK_SIZE and BATCH_SIZE. If you reduce
  #     BLOCK_SIZE to 1 it's a bigram model !-->


if __name__ == "__main__":
  main()