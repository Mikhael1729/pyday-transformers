import torch

from data import Data, Datasets, DatasetSplit
from model import BigramLanguageModel
from variables import (BATCH_SIZE, EVALUATION_INTERVAL, EVALUATION_ITERATIONS,
                       LEARNING_RATE, SAMPLING_SIZE, TRAINING_STEPS, device)

torch.manual_seed(1337) # For reproducibility


def main():
  """
  3. Basic model + basic level of interaction:

  This is the same code as the previous iteration, but with the added linear layer
  in the architecture to map the features of the learned look-up table into scores
  for next token prediction.

  These changes don't improve the performance of the model.
  """
  global device

  # Load the raw data
  data = Data("./input.txt")

  # Encode the data using the stoi mapping and store the result in a tensor.
  datasets = Datasets(torch.tensor(data.encode(data.text), dtype=torch.long, device=device))
  
  # Build the n-gram (arbitrary context lenght `n`) language model.
  model = BigramLanguageModel()
  model.to(device)

  # Train model
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
  for step in range(TRAINING_STEPS):
    # Evaluate loss every once in  while using an empiric based evaluation function
    if step % EVALUATION_INTERVAL == 0:
      losses = estimate_loss(model, datasets, EVALUATION_ITERATIONS)
      print(f"step {step}: train loss {losses[DatasetSplit.train]:.4f}, val loss {losses[DatasetSplit.val]:.4f}")
      
    # Sample a batch of data
    Xb, Yb = datasets.get_batch(split=DatasetSplit.train, batch_size=BATCH_SIZE)

    # Evaluate loss and optimize the model
    _, loss = model(Xb, Yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  # Initialize first_token as a zero representing the "\n" character with shape (b, c), 
  # where b is the batch size (1) and c is the context length (1), as it's a single letter
  first_token = torch.zeros((1, 1), dtype=torch.long, device=device)

  # Generate 100 other tokens starting with first_token
  generated_tokens = model.generate(first_token, SAMPLING_SIZE)

  # Extract the tokens from the first and only batch and convert it into a list of integers
  generated_tokens = generated_tokens[0].tolist()

  print(data.decode(generated_tokens))


@torch.no_grad()
def estimate_loss(model: BigramLanguageModel, datasets: Datasets, evaluation_iterations: int) -> dict[DatasetSplit, float]:
  """
  This function allows you to get what I would call a more empirically
  based measurement of the loss. It’s an average of 200 random evaluations
  of the model.
  """
  out: dict[DatasetSplit, float] = {}
  
  # Set model to evaluation mode. Done because some layers may behave
  # differently deppending on the mode
  model.eval()

  for split in [DatasetSplit.train, DatasetSplit.val]:
    losses = torch.zeros(evaluation_iterations)

    for k in range(evaluation_iterations):
      Xb, Yb = datasets.get_batch(split)
      _, loss = model(Xb, Yb)

      losses[k] = loss.item()
    
    out[split] = losses.mean()

  # Set model to training mode
  model.train()

  return out


if __name__ == "__main__":
  main()