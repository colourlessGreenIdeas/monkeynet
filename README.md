## Background:
This is not a serious deep learning package- although it works quite well for
fully connected sigmoid activated networks.

The genesis of this was a way for me to get back into coding quickly and with
an example I thought would blend math, algos, and python together. And I wanted to build this with only
one dependency- numpy.

As the name implies, I was simply monkeying around. Hence, monkeynet.

**TODO**: a whole bunch!
- add softmax
- maybe add a convolution option
- RelU activation
- additional loss function [probably an entire class]

## Monkeynet: Order of operations:
- Create Layers
- Create Network from Layers
- Create Optimizer
- Create a DataLoader
- Create a crappy backprop
- Create a good backprop
- Test against MNIST

**Note:**
Get the the mnist data from here:
http://yann.lecun.com/exdb/mnist/    

Make sure to gzip before using, like so:
gzip -d mnist.pkl.gz

then <code>python3 mnist_example.py</code>
