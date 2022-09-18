import numpy as np
from network import Network

class Optimizer:
    """Optimize the weights using various methods.
    Here we implement the SGD for now.
    We can easily add other optimizers but SGD is the obvious and easiest one.
    """
    def __init__(self) -> None:
        pass
    
    def update(self) ->None:
        raise NotImplementedError
    

class SGD(Optimizer):
    
    def __init__(self, learning_rate: float, network: Network) -> None:
        """All you have to do is suply the learning rate and the network itself
        It does the rest"""
        self.eta = learning_rate
        self.network = network
        
    def update(self) -> None:
        """
        1. Pass inputs intto the network
        2. For all the inputs passed, add up the weight and bias grads for each layer in the network
        3. adjust the weight and bias by the above
        
        w_l = w_l - eta*(1/m)*sum(grad_w)
        b_l = b_l - eta*(1/m)*sum(grad_b)
        """
        
        for layer in self.network.layers:
            # clear the list of each layers' gradients after each update
            layer.weights = layer.weights - (self.eta * (1/len(layer.grad_ws)) * np.sum(layer.grad_ws,axis=0))
            layer.grad_ws = []
            
            layer.biases = layer.biases - self.eta * (1/len(layer.grad_bs)) * np.sum(layer.grad_bs,axis=0)
            layer.grad_bs = []
            