# objectives:
# -implement feedforward
# -implement backpropagation
# -set up autoencoder
# -do exercise from ufldl tutorial

import math
import numpy as np
from scipy.misc import derivative

def sigmoid(x):
    return 1. / (1. + math.exp(-x))

class ActivationFunction:
    def __init__(self, activation_func):
        self.activation_func = activation_func
    def eval(self, x):
        return self.activation_func(x)
    def deriv(self, x):
        if self.activation_func == sigmoid:
            sigmoid_val = sigmoid(x)
            return sigmoid_val * (1 - sigmoid_val)
        else:
            return derivative(self.activation_func, x)
	
class NeuralNetwork:
    def __init__(self, num_nodes_per_layer, activation_func=sigmoid):
        self.num_nodes_per_layer = num_nodes_per_layer
        self.activation_func = ActivationFunction(activation_func)
        
    def check_inputs(self, input, biases, weights):
        assert len(input) == self.num_nodes_per_layer[0]
        assert len(biases) == len(self.num_nodes_per_layer) - 1
        assert len(weights) == len(self.num_nodes_per_layer) - 1
        for i, bias in enumerate(biases):
            assert len(bias) == self.num_nodes_per_layer[i + 1]
        for i, weight in enumerate(weights):
            assert weight.shape == \
                   (self.num_nodes_per_layer[i + 1], self.num_nodes_per_layer[i])

    def feedforward(self, input, biases, weights):
        self.check_inputs(input, biases, weights)
        activation = input
        pre_activations = []
        vectorized_activation = np.vectorize(self.activation_func.eval)
        for bias, weight in zip(biases, weights):
            with_one = np.insert(activation, 0, 1.)
            bias_and_weight = np.hstack((np.array([bias]).T, weight))
            pre_activation = bias_and_weight.dot(with_one)
            pre_activations.append(pre_activation)
            activation = vectorized_activation(pre_activation)
        return activation, pre_activations
        
    def backpropagate(self, inputs, outputs, biases, weights):
        bias_deriv = biases - biases
        for input, output in zip(inputs, outputs):
            activation, pre_activations = self.feedforward(input, biases, weights)
            vectorized_activation_deriv = np.vectorize(self.activation_func.deriv)
            pre_activation_derivs = \
                [vectorized_activation_deriv(pre_activation) \
                     for pre_activation in pre_activations \
                ]
            cost_pre_activation_deriv = \
                (output - activation) * pre_activation_derivs[-1]
            cost_pre_activation_derivs = [cost_pre_activation_deriv]
            for pre_activation_deriv, weight in zip(pre_activation_derivs[:-1], weights):
                cost_pre_activation_deriv = \
                    weight.T.dot(cost_pre_activation_deriv) * pre_activation_deriv
                cost_pre_activation_derivs.insert(0, cost_pre_activation_deriv)
            bias_deriv = bias_deriv + cost_pre_activation_derivs
        
if __name__ == '__main__':
    x = NeuralNetwork([2, 3, 4])
    x.feedforward(np.array([.2, .3]), \
                  np.array([np.array([.3, .4, .5]), np.array([.7, .8, .9, .10])]), \
                  [np.array([[.4, .2], [.6, .3], [.9, .2]]), \
                   np.array([[.3, .5, .2], [.7, .4, .2], [.4, .6, .2], [.2, .3, .6]])
                  ]
                 )
    x.backpropagate([np.array([.2, .3]), np.array([.4, .7])],
                    [np.array([.9, .1, .2, .3]), np.array([.1, .2, .7, .4])],
                    np.array([np.array([.3, .4, .5]), np.array([.7, .8, .9, .10])]), \
                    [np.array([[.4, .2], [.6, .3], [.9, .2]]), \
                    np.array([[.3, .5, .2], [.7, .4, .2], [.4, .6, .2], [.2, .3, .6]])
                    ]
                    )
    print 3