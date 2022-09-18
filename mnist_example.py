import pickle
import numpy as np
from numpy import ndarray as Tensor

from layer import FullyConnected
from network import Network
from optimizer import SGD
from dataloader import DataLoader
from trainer import Trainer

def label_to_vector(label: Tensor) -> Tensor:
    """We will conver the individual number classes, 
    like 5,4,3,7,8... in the downbloaded mnsit
    database into a (10,1) vector X with X[index] = 1 
    where index is the number itself"""
    vec = np.zeros((10,1))
    vec[label] = 1
    return vec


# Get the data from here:
# http://yann.lecun.com/exdb/mnist/    

# Make sure to gzip before using, like so:
# gzip -d mnist.pkl.gz

with open('mnist.pkl', 'rb') as f:
    trd, vd, tsd = pickle.load(f, encoding="latin1")

mnist_inputs = np.array([np.reshape(tr_i, (784, 1)) for tr_i in trd[0]])
mnist_outputs = np.array([label_to_vector(tr_o) for tr_o in trd[1]])

mnist_test_inputs = np.array([np.reshape(ts_i, (784, 1)) for ts_i in tsd[0]])
mnist_test_outputs = np.array([label_to_vector(ts_o) for ts_o in tsd[1]])

mnist_validation_inputs = np.array([np.reshape(vd_i, (784, 1)) for vd_i in vd[0]])
mnist_validation_outputs = np.array([label_to_vector(vd_o) for vd_o in vd[1]])


# We will have two layers in our neural network
fc1 = FullyConnected(784,50)
fc2 = FullyConnected(50,10)

# Create the network
nn = Network([fc1, fc2])

# Ready the optimizer:
batch_size = 30

sgd = SGD(learning_rate=3,network=nn)

trainer = Trainer()
trainer.train(network=nn,
              inputs=mnist_inputs,
              outputs=mnist_outputs,
              optimizer=sgd,
              batch_size=batch_size,
              num_epochs=30
             )
