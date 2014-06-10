import numpy as np
from images import display_image_grid
from learning import LearningMethod
from mnist import load_mnist
from neural_network import NeuralNetwork, SparsityParams

''' Because data is between 0 and 1, don't need to normalize to put into
    autoencoder.
    
    However, it is interesting to note that if you did normalize using
    normalize_image_slices(), the sparsity calc blows up.
    
    What seems to happen is that because the handwriting pixels are generally
    bimodal (all black or all white), the data has a large variance, and the
    normalization ends up shrinking the data to be quite close to 0.5.
    
    Then, in the inner layer activation calc, a given neuron's pre-activation
    is basically going to be the sum of all these 0.5's times a bunch of
    random normals (the initial guesses for the weights).
    
    With a large number of random normals, you're going to get a large
    variance, so some of the pre-activations end up being very negative or
    very positive, which the sigmoid then converts into 0's or 1's for
    virtually all the input data, which is obviously pretty bad for the
    sparsity calc. '''
def sparse_autoencoder_mnist():
    training, test = load_mnist()
        
    data = training[0][0:10000]
    autoencoder_mnist_network = \
        NeuralNetwork([784, 196, 784], \
                      regularization=0.003, \
                      sparsity_params=SparsityParams(0.1, 3.) \
                     )
    autoencoder_mnist_network.train([data, data], \
                                    [], \
                                    1., \
                                    1, \
                                    1, \
                                    learning_method=LearningMethod('L-BFGS-B', {'max_iter' : 400}) \
                                   )
    weight = autoencoder_mnist_network.weights[0]
    display_image_grid(weight, 28, 14)

if __name__ == '__main__':
    sparse_autoencoder_mnist()
