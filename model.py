import torch
import torch.nn as nn
from torch.nn import functional as F

from variables import BLOCK_SIZE, N_EMBEDDINGS, VOCABULARY_SIZE, device, DROPOUT


class Block(nn.Module):
  """
  The transformer block contains a multi-head attention module and a feedforward network,
  effectively encapsulating within a single entity both the contextual information provided
  by the self-attention heads and the analysis or computation performed by the feedforward
  network.
  """
  def __init__(self, n_embeddings, n_heads):
    super().__init__()

    head_size = n_embeddings // n_heads

    self.self_attention = MultiheadAttention(n_heads, head_size)
    self.feedforward = Feedforward(n_embeddings)

  def forward(self, encoded_words: torch.Tensor):
    # Performs the extraction of contextual information
    encoded_words = encoded_words + self.self_attention(encoded_words)

    # Applies the analysis or computation used for next token prediction
    encoded_words = encoded_words + self.feedforward(encoded_words)

    return encoded_words


class Feedforward(nn.Module):
  """
  Applies a nonlinear transformation to the embeddings to refine and enrich their
  representations
  """
  def __init__(self, n_embeddings: int):
    super().__init__()

    # The 4 in the dimensionality is done as indicated in the relative dimensions in
    # the paper Attention Is All You Need
    self.network = nn.Sequential(
      nn.Linear(n_embeddings, 4 * n_embeddings),
      nn.ReLU(),
      nn.Linear(4 * N_EMBEDDINGS, N_EMBEDDINGS) # Projection layer maps the output into the same shape as the residual pathway to perform the sum operation
    )

  def forward(self, encoded_words: torch.Tensor):
    return self.network(encoded_words)


class MultiheadAttention(nn.Module):
  """
  Multiple heads of self-attention in parallel
  """
  def __init__(self, num_heads: int, head_size: int):
    super().__init__()
    # Each self-attention head learns to focus on different types of contextual
    # relationships between the current token (nᵗʰ position) and the tokens in
    # its available context, in order to estimate which of them carry more
    # weight for predicting the next token.
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    # The projection linearly combines the outputs of all self-attention heads,
    # bringing them back to the model's embedding dimension so they can be added
    # to the residual stream. It also integrates the information extracted by
    # the heads into a unified representation.
    self.projection = nn.Linear(N_EMBEDDINGS, N_EMBEDDINGS)

  def forward(self, encoded_words: torch.Tensor):
    # The concatenation is done in the features or channel dimension
    output = torch.cat([head(encoded_words) for head in self.heads], dim=-1)

    # The projection is the linear transformation of the output of the previous layer
    output = self.projection(output)

    return output


class Head(nn.Module):
  """
  Single head of self-attention
  """
  def __init__(self, head_size: int):
    super().__init__()

    self.head_size = head_size

    # Projects input embeddings into key space used to measure similarity with queries
    self.key = nn.Linear(N_EMBEDDINGS, self.head_size, bias=False)

    # Projects input embeddings into query space to compute attention scores against keys
    self.query = nn.Linear(N_EMBEDDINGS, self.head_size, bias=False)

    # Projects input embeddings into value space (the actual information to be aggregated)
    self.value = nn.Linear(N_EMBEDDINGS, self.head_size, bias=False)

    # Lower-triangular mask to prevent tokens from attending to future positions
    self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

  def forward(self, encoded_words: torch.Tensor):
    # batch size, context size, features length
    _, c, _ = encoded_words.shape

    # Compute the weight matrices
    key = self.key(encoded_words) # (b, c, f)
    query = self.query(encoded_words) # (b, c, f)

    # Compute the attention scores
    scores = query @ key.transpose(-2, -1) * self.head_size**-0.5 # (b, c, f) @ (b, f, c) -> (b, c, c)
    scores = scores.masked_fill(self.tril[:c, :c] == 0, float('-inf'))
    scores = F.softmax(scores, dim=-1)

    # Compute the relationship strenght between the given values and the scores
    value = self.value(encoded_words) # (b, c, f)
    attention = scores @ value # (b, c, c) @ (b, c, f) -> (b, c, f)

    return attention


class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    
    # Maps tokens to learned vectors; tokens with similar next-token patterns
    # get similar embeddings.
    self.token_embeddings = nn.Embedding(VOCABULARY_SIZE, N_EMBEDDINGS)

    # Encodes the position of each token in the secuence.
    self.position_embeddings = nn.Embedding(BLOCK_SIZE, N_EMBEDDINGS)

    # Performs the extraction of contextual information and the analysis of it
    self.blocks = nn.Sequential(
      Block(N_EMBEDDINGS, n_heads=4),
      Block(N_EMBEDDINGS, n_heads=4),
      Block(N_EMBEDDINGS, n_heads=4),
    )

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
    b, c = encoded_words.shape

    # Obtain the scores to determine the most likely next token
    token_embeddings = self.token_embeddings(encoded_words) # (b, c, N_EMBEDDINGS).

    # Get embeddings that encode the position (0 to c−1) of each token in the sequence.
    position_embeddings = self.position_embeddings(torch.arange(c, device=device)) # (c, f)

    # Agregate the embeddings into a single learnable set of features
    combined_embeddings = token_embeddings + position_embeddings # (b, c, f)

    # Includes the contextual information of the input into combined_embeddings
    combined_embeddings = self.blocks(combined_embeddings)

    # Decode the given features to a series of scores for next token prediction
    logits = self.lm_head(combined_embeddings) # (b, c, VOCABULARY_SIZE)

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
    
