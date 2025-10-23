import torch
import torch.nn as nn
from torch.nn import functional as F

from variables import BLOCK_SIZE, N_EMBEDDINGS, VOCABULARY_SIZE


class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    
    # Maps tokens to learned vectors; tokens with similar next-token patterns
    # get similar embeddings.
    self.token_embeddings = nn.Embedding(VOCABULARY_SIZE, N_EMBEDDINGS)

    # Language modeling head (output layer). It converts the high-dimensional
    # embeddings into a probability distribution over the vocabulary to predict
    # the next token
    self.lm_head = nn.Linear(N_EMBEDDINGS, VOCABULARY_SIZE)

  def forward(self, encoded_words: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    - encoded_words. A tensor containing the encoded input words. It's expected to be 
      of shape(B, C), where B is the batch size and C is the context lenght

    - targets. A tensor containing the the next character for each of the C
      combination of words in encoded_words. For this it's expected to be of shape
      (B, C), same as encoded_words.
    """
    # Obtain the scores to determine the most likely next token
    token_embeddings = self.token_embeddings(encoded_words) # (b, c, N_EMBEDDINGS)

    # Decode the given features to a series of scores for next token prediction
    logits = self.lm_head(token_embeddings) # (b, c, VOCABULARY_SIZE)

    # Inference is requested
    if targets is None:
      return logits, None

    # Rearange logits to comply with the requirements of F.cross_entropy.
    # b is the size of the batch dimension
    # c is the size of the context or sample temporal dimension
    # f is the number of features or classes
    b, c, f = logits.shape
    logits = logits.view(b * c, f)
    targets = targets.view(b * c)

    # Initially this value should be -ln(1 / VOCABULARY_SIZE)
    loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, encoded_words: torch.Tensor, max_new_tokens: list[int]):
    for _ in range(max_new_tokens):
      # Crop encoded words to only accept a maximum of BLOCK_SIZE
      encoded_words_cropped = encoded_words[:, -BLOCK_SIZE:]

      # (b, c, f) The execution of __call__ internally runs the forward function which return the logits of the model 
      logits, _ = self(encoded_words_cropped)
      
      # (b, f) Query the features of the last generated token to predict the next one
      last_encoded_token_logits = logits[:, -1, :]

      # (b, f) Create a probability distribution from the logits of the last generated token
      probabilities = F.softmax(last_encoded_token_logits, dim=-1)

      # (b, 1) Sample from the probabilities: next_token correspons one of the 65 encoded characters 
      next_token = torch.multinomial(probabilities, num_samples=1)

      # (b, c + 1) Concatenate new generated token to the generation tensor
      encoded_words = torch.cat((encoded_words, next_token), dim=1)
    
    return encoded_words
    