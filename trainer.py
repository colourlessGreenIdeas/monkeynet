import numpy as np
from numpy import ndarray as Tensor
from network import Network
from optimizer import Optimizer
from dataloader import DataLoader

class Trainer:
    
    def __init__(self) -> None:
        """Use this class to start the training process
        supply the network, the inputs, outputs, the optimizer, 
        the batch size, the number of epochs, the """
        pass
    
    def train(self,
              network: Network,
              inputs: Tensor,
              outputs: Tensor,
              optimizer: Optimizer,
              batch_size,
              num_epochs,
             ) -> None:
        
        # Start here for training:
        for i in range(num_epochs):
            
            loss_epoch = 0.0
            acc_epoch = 0.0
            
            for batch in DataLoader(batch_size).get_batches(inputs, outputs):
                for inp, out in zip(batch[0], batch[1]):
                    network.forward_pass(inp)
                    network.backprop(inp, out)
                    acc_epoch += network.accuracy(network.forward_pass(inp), out)
                optimizer.update()
                    
            print("epoch : {} | acc : {}".format(i, 100*acc_epoch/len(inputs)))
            
        
