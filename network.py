import numpy as np
from numpy import ndarray as Tensor
from typing import Sequence
from layer import FullyConnected
class Network:
    """Create the network. It should loop through
    the layers in the network and perform both the forward pass
    and the backwards pass, a.k.a. the backpropagation.
    
    also contains simple methods to calculate the sigmoid and its
    derivative like in the layer.
    
    TODO: Perhaps the loss and accuracy can be moved elsewhere?
    """
    
    def __init__(self, layers: Sequence[FullyConnected])->None:
        self.layers = layers
        self.activations = []
        self.zs = []
        
    def forward_pass(self, inputs: Tensor) -> Tensor:
        self.activations = []
        self.activations.append(inputs)
        self.zs = []
        for layer in self.layers:
            inputs = layer.activate(inputs)
            self.zs.append(layer.z)
            self.activations.append(inputs)
        return inputs

        
    def backprop(self, inputs: Tensor, outputs: Tensor) -> None:

        self.delta = (self.activations[-1] - outputs) * self.sigmoid_deriv(self.zs[-1])
        self.layers[-1].delta = self.delta
        self.layers[-1].grad_b = self.delta
        self.layers[-1].grad_w = self.delta @ self.activations[-2].T
        
        self.layers[-1].grad_bs.append(self.layers[-1].grad_b)
        self.layers[-1].grad_ws.append(self.layers[-1].grad_w)

        for l in range(2, len(self.layers)+1):

            self.delta = (self.layers[-l+1].weights.T @ self.delta) * self.sigmoid_deriv(self.zs[-l])
            
            self.layers[-l].grad_b = self.delta
            self.layers[-l].grad_bs.append(self.layers[-l].grad_b)
            
            self.layers[-l].grad_w = self.delta @ self.activations[-l-1].T
            self.layers[-l].grad_ws.append(self.layers[-l].grad_w)
                                                                                      
                                                                                
            
    def sigmoid(self, z: Tensor) -> Tensor:
        return 1/(1 + (np.exp(-z)))
    
    def sigmoid_deriv(self, z: Tensor) -> Tensor:
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def loss(self, predicted: Tensor, actual: Tensor)-> float:
        
        pred_idx = np.argmax(predicted)
        actual_idx = np.argmax(actual)
        
        pred_arr = np.zeros(predicted.shape)
        pred_arr[pred_idx] = 1
        
        actual_arr = np.zeros(predicted.shape)
        actual_arr[actual_idx] = 1
        
        return np.sum((actual_arr - pred_arr) ** 2)
    
    def accuracy(self, predicted: Tensor, actual: Tensor)-> float:
        
        pred_idx = np.argmax(predicted)
        actual_idx = np.argmax(actual)
        
        return 1 if pred_idx == actual_idx else 0
        
        