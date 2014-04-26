# objectives:
# -implement feedforward
# -implement backpropagation
# -set up autoencoder
# -do exercise from ufldl tutorial

import math
import numpy as np

def sigmoid(x):
    return 1. / (1. + math.exp(-x))
    
sigmoid_vec = np.vectorize(sigmoid)

class NeuralNetwork:
    def __init__(self, num_nodes_per_layer):
        self.num_nodes_per_layer = num_nodes_per_layer
        
    def check_inputs(self, input_row, biases, weights):
        assert len(input_row) == self.num_nodes_per_layer[0]
        assert len(biases) == len(self.num_nodes_per_layer) - 1
        assert len(weights) == len(self.num_nodes_per_layer) - 1
        for i, bias in enumerate(biases):
            assert len(bias) == self.num_nodes_per_layer[i + 1]
        for i, weight in enumerate(weights):
            assert weight.shape == \
                   (self.num_nodes_per_layer[i + 1], self.num_nodes_per_layer[i])

    def feedforward(self, input_row, biases, weights):
        self.check_inputs(input_row, biases, weights)
        activation = input_row
        for bias, weight in zip(biases, weights):
            with_one = np.insert(activation, 0, 1.)
            bias_and_weight = np.hstack((np.array([bias]).T, weight))
            activation = np.vectorize(sigmoid)(bias_and_weight.dot(with_one))
            print activation
        return activation
        
if __name__ == '__main__':
    x = NeuralNetwork([2, 3, 4])
    x.feedforward(np.array([.2, .3]), \
                  [np.array([.3, .4, .5]), np.array([.7, .8, .9, .10])], \
                  [np.array([[.4, .2], [.6, .3], [.9, .2]]), \
                   np.array([[.3, .5, .2], [.7, .4, .2], [.4, .6, .2], [.2, .3, .6]])
                  ]
                 )
    print 3