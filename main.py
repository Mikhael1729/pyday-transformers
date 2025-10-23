import torch

from data import Data, Datasets
from model import BigramLanguageModel
from variables import device

torch.manual_seed(1337) # For reproducibility


def main():
  """
  2. Basic model:

  The most basic model uses a list of embeddings to represent the semantic meaning
  of different characters in order to predict the next one.

  In this code snippet, you can see how to use the model to generate data.  
  Notice that the model is currently generating gibberish, but later we will make
  the architecture more elaborate to produce better predictions.
  """
  global device

  # Load the raw data
  data = Data("./input.txt")
  
  # Build the n-gram (arbitrary context lenght `n`) language model.
  model = BigramLanguageModel()
  model.to(device)

  # Initialize first_token as a zero representing the "\n" character with shape (b, c), 
  # where b is the batch size (1) and c is the context length (1), as it's a single letter
  first_token = torch.zeros((1, 1), dtype=torch.long, device=device)

  # Generate 100 other tokens starting with first_token
  generated_tokens = model.generate(first_token, 100)

  # Extract the tokens from the first and only batch and convert it into a list of integers
  generated_tokens = generated_tokens[0].tolist()

  print(data.decode(generated_tokens))


if __name__ == "__main__":
  main()