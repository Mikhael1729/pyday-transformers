import torch
from enum import Enum
from typing import Tuple
from variables import BATCH_SIZE, BLOCK_SIZE

class DatasetSplit(str, Enum):
  train = "train"
  val = "val"

class Datasets:
  def __init__(self, data: torch.Tensor):
    ninety_percent = int(0.9 * len(data))

    self.train = data[:ninety_percent]
    self.val = data[ninety_percent:]

  def get_batch(self, split: DatasetSplit, batch_size: int = BATCH_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    # Select the appropiate dataset to get a batch from
    data = self.train if split is DatasetSplit.train else self.val
    
    # Generate random starting positions to construct the context data.
    # BLOCK_SIZE is subtracted to avoid generating a context of lenght BLOCK_SIZE
    # in the last character, which would produce an index out of range error
    indices = torch.randint(len(data) - BLOCK_SIZE, (batch_size, ))

    x = torch.stack([data[index:index+BLOCK_SIZE] for index in indices])
    y = torch.stack([data[index+1:index+BLOCK_SIZE+1] for index in indices])

    return x, y
    

class Data:
  def __init__(self, file_path: str):
    self.text = get_text(file_path)
    self.unique_chars = sorted(list(set(self.text)))
    self.vocabulary_size = len(self.unique_chars)
    self.stoi = { c: i for i, c in enumerate(self.unique_chars)}
    self.itos = { i: c for i, c in enumerate(self.unique_chars)}

  def encode(self, string: str) -> list[int]:
    return [ self.stoi[s] for s in string ]

  def decode(self, encoded_string: list[int]) -> str:
    return ''.join([self.itos[s] for s in encoded_string ])

def get_text(file_path: str) -> str:
  with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

  return text
    
