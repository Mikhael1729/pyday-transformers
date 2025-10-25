import torch

"""
Training variables
"""
# Number of samples per gradiente update
# Increased value from 32
BATCH_SIZE = 64

# Number of optimization steps in the given batch
TRAINING_STEPS = 5000

# Number of steps to run before computing the evaluation data
EVALUATION_INTERVAL = 500

# Number of inference steps for the given evaluation batch
EVALUATION_ITERATIONS = 200

# It's the size of the update after a single iteration of gradient descente
# The learning rate went from 1e-3 to this value to make smaller changes during optimization
LEARNING_RATE = 3e-4

# The number of characters to sample from the model
SAMPLING_SIZE = 10000

"""
Data variables
"""
# The context lenght of the model
BLOCK_SIZE = 256

# The number of unique characters recognized by the model (See the Data class for more info)
VOCABULARY_SIZE = 65 

# Size of the embeddings used to encode various kinds of information related to the data
# Increased value from 32
N_EMBEDDINGS = 384

"""
Network configuration
"""
# Number of self attention blocks
N_LAYERS = 6

# Number of self attention heads
N_HEAD = 6 # Because N_EMBEDDINGS // N_HEAD = 384 / 6 = 64 parameters on each head

# The percentage of neurons to trim from the network for regularization on every forward-backward pass
DROPOUT = 0.2

# Device name (to choose the best machine to run the code)
device = None

def get_device() -> str:
  global device

  if device is not None:
    return device

  if torch.backends.cuda.is_built():
    device = 'cuda'
  elif torch.backends.mps.is_available():
    device = 'mps'
  else:
    device = 'cpu'

  return device

device = get_device()
