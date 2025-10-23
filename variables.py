import torch

"""
Training variables
"""
# Number of samples per gradiente update
BATCH_SIZE = 32

# Number of optimization steps in the given batch
TRAINING_STEPS = None

# Number of steps to run before computing the evaluation data
EVALUATION_INTERVAL = None

# Number of inference steps for the given evaluation batch
EVALUATION_ITERATIONS = None

# It's the size of the update after a single iteration of gradient descente
LEARNING_RATE = None

"""
Data variables
"""
# The context lenght of the model
BLOCK_SIZE = 8

# The number of unique characters recognized by the model (See the Data class for more info)
VOCABULARY_SIZE = 65 

# Size of the embeddings used to encode various kinds of information related to the data
N_EMBEDDINGS = 65

"""
Network configuration
"""
# Number of self attention blocks
N_LAYERS = None 

# Number of self attention heads
N_HEAD = None

# The percentage of neurons to trim from the network for regularization
DROPOUT = None

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
