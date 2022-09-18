import numpy as np
from numpy import ndarray as Tensor

class Layer:
    """Creates a layer.
    Layers can be different types.
    Here we will create the full connected layer
    with the sigmoid activation
    """
    def __init__(self)->None:
        pass
    
    def feedforward(self, inputs):
        raise NotImplementedError
    
    def backprop(self, inputs, outputs):
        return NotImplementedError


class FullyConnected(Layer):
    """The full connected layer created the weights and biases.
    It also stores the gradients associated to each weights since
    these gradients are needed to update the weight in backpropagation. 
    """
    def __init__(self, input_size, output_size)->None:
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.randn(self.output_size, self.input_size)
        self.biases = np.random.randn(self.output_size, 1)
        
        self.grad_w = np.zeros(self.weights.shape)
        self.grad_b = np.zeros(self.biases.shape)
        
        self.grad_ws = []
        self.grad_bs = []
        
    def activate(self, inputs: Tensor)->Tensor:
        
        self.z = self.weights @ inputs + self.biases
        self.activation = self.sigmoid(self.z)
        return self.activation
    
    
    def sigmoid(self, z: Tensor) -> Tensor:
        return 1/(1 + (np.exp(-z)))
    
    def sigmoid_deriv(self, z: Tensor) -> Tensor:
        return self.sigmoid(z) * (1 - self.sigmoid(z))