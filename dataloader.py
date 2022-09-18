import numpy as np
from numpy import ndarray as Tensor
from typing import Iterator, Tuple

class DataLoader:
    """Provides methods to suffle and batch data.
    We could probably extend it to handle other tasks 
    such as comverting datasets into desired tensor-like structures
    But that's for another day"""
    
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
    
    def get_batches(self, inputs: Tensor, outputs: Tensor) -> Iterator[Tuple]:
        """For all the inputs, shuffle the indicies and then use the index + batch_size 
        to create batches to pass to the gradient descent algorithm (or others).
        """
        idx = np.arange(0, len(inputs), self.batch_size)
        np.random.shuffle(idx)

        shuffled_batches = [(inputs[k:k+self.batch_size], outputs[k:k+self.batch_size]) for k in idx]
        return shuffled_batches
    