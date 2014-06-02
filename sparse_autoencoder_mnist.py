import numpy as np
from images import display_image_grid, generate_random_image_slice, normalize_image_slices
from mnist import load_mnist
from neural_network import LearningMethod, NeuralNetwork, SparsityParams

def sparse_autoencoder_mnist():
    training, validation, test = load_mnist()
    # figured out the right reshape to use in generate_random_image_slice
    # by trial and error..
    normalized_image_slices = normalize_image_slices(training[0][0:10000])
    autoencoder_mnist_network = \
        NeuralNetwork([784, 196, 784], \
                      regularization=0.003, \
                      sparsity_params=SparsityParams(0.1, 3.) \
                     )
    autoencoder_mnist_network.train([normalized_image_slices, normalized_image_slices], \
                                    [], \
                                    [], \
                                    1., \
                                    1, \
                                    1, \
                                    learning_method=LearningMethod('L-BFGS-B', {'max_iter' : 1}) \
                                   )
    # weight = autoencoder_mnist_network.weights[0]
    # display_image_grid(weight, 28, 14)

if __name__ == '__main__':
    sparse_autoencoder_mnist()
