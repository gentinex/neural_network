# TODO:
# -compare / contrast implementation to nielsen's book
# -run on handwritten digits
# -add regularization
# -set up autoencoder
# -do exercise from ufldl tutorial

import math
import numpy as np
from numpy.random import randn, seed
from scipy.misc import derivative

EPSILON = 1e-10

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
        
    def check_input(self, input, biases, weights):
        assert len(input) == self.num_nodes_per_layer[0]
        assert len(biases) == len(self.num_nodes_per_layer) - 1
        assert len(weights) == len(self.num_nodes_per_layer) - 1
        for i, bias in enumerate(biases):
            assert len(bias) == self.num_nodes_per_layer[i + 1]
        for i, weight in enumerate(weights):
            assert np.array(weight).shape == \
                   (self.num_nodes_per_layer[i + 1], self.num_nodes_per_layer[i])

    def feedforward(self, input, biases, weights):
        self.check_input(input, biases, weights)
        vectorized_activation = np.vectorize(self.activation_func.eval)
        activations = []
        activation = input
        activations.append(activation)
        pre_activations = []
        for bias, weight in zip(biases, weights):
            with_one = np.insert(activation, 0, 1.)
            bias_and_weight = np.hstack((np.array([bias]).T, weight))
            pre_activation = bias_and_weight.dot(with_one)
            pre_activations.append(pre_activation)
            activation = vectorized_activation(pre_activation)
            activations.append(activation)
        return activation, activations, pre_activations
        
    def backpropagate(self, inputs, outputs, biases, weights):
        bias_derivs = [np.zeros(bias.shape) for bias in biases]
        weight_derivs = [np.zeros(weight.shape) for weight in weights]
        for input, output in zip(inputs, outputs):
            activation, activations, pre_activations = \
                self.feedforward(input, biases, weights)
            vectorized_activation_deriv = np.vectorize(self.activation_func.deriv)
            pre_activation_derivs = \
                [vectorized_activation_deriv(pre_activation) \
                     for pre_activation in pre_activations \
                ]
            cost_pre_activation_deriv = \
                (activation - output) * pre_activation_derivs[-1]
            cost_pre_activation_derivs = [cost_pre_activation_deriv]
            for pre_activation_deriv, weight in \
                reversed(zip(pre_activation_derivs[:-1], weights[1:])):
                cost_pre_activation_deriv = \
                    weight.T.dot(cost_pre_activation_deriv) * pre_activation_deriv
                cost_pre_activation_derivs.insert(0, cost_pre_activation_deriv)
            current_weight_derivs = \
                [np.outer(cost_pre_activation_deriv_b, activation_b) \
                 for cost_pre_activation_deriv_b, activation_b in \
                     zip(cost_pre_activation_derivs, activations[:-1])
                ]
            bias_derivs = [bias_deriv + cost_pre_activation_deriv for \
                               bias_deriv, cost_pre_activation_deriv in \
                               zip(bias_derivs, cost_pre_activation_derivs) \
                          ]
                            
            weight_derivs = [weight_deriv + current_weight_deriv for \
                                 weight_deriv, current_weight_deriv in \
                                 zip(weight_derivs, current_weight_derivs) \
                            ]
        return bias_derivs, weight_derivs
        
    def cost(self, inputs, outputs, biases, weights):
        net_cost = 0
        for input, output in zip(inputs, outputs):
            predicted_output, _, _ = self.feedforward(input, biases, weights)
            net_cost += sum(0.5 * (predicted_output - output) ** 2.)
        return net_cost
        
    def cost_deriv(self, inputs, outputs, biases, weights):
        bias_derivs = [np.zeros(bias.shape) for bias in biases]
        weight_derivs = [np.zeros(weight.shape) for weight in weights]
        base_cost = self.cost(inputs, outputs, biases, weights)
        for i, bias in enumerate(biases):
            for j, bias_elt in enumerate(bias):
                biases[i][j] += EPSILON
                bias_derivs[i][j] = \
                    (self.cost(inputs, outputs, biases, weights) - base_cost) / EPSILON
                biases[i][j] -= EPSILON
        for i, weight in enumerate(weights):
            for j, weight_row in enumerate(weight):
                for k, weight_col in enumerate(weight_row):
                    weights[i][j][k] += EPSILON
                    weight_derivs[i][j][k] = \
                        (self.cost(inputs, outputs, biases, weights) - base_cost) / EPSILON
                    weights[i][j][k] -= EPSILON
        return bias_derivs, weight_derivs

    def train(self, inputs, outputs):
        # learning rate is quite important - note e.g. that linear seems to
        # require much smaller rates than sigmoid to properly converge
        learning_rate = 1.
        seed(1)
        biases = []
        weights = []
        for i, num_nodes in enumerate(self.num_nodes_per_layer[1:]):
            biases.append(randn(num_nodes))
            weights.append(randn(num_nodes, self.num_nodes_per_layer[i]))
        while True:
            bias_derivs, weight_derivs = \
                self.backpropagate(inputs, outputs, biases, weights)
            if all(all(abs(bias_deriv_elt) < EPSILON \
                       for bias_deriv_elt in bias_deriv) \
                       for bias_deriv in bias_derivs) \
               and \
               all(all(all(abs(weight_deriv_elt) < EPSILON \
                           for weight_deriv_elt in weight_deriv_row) \
                           for weight_deriv_row in weight_deriv) \
                           for weight_deriv in weight_derivs):
                break
            biases = [bias - learning_rate * bias_deriv for bias, bias_deriv \
                          in zip(biases, bias_derivs) \
                     ]
            weights = [weight - learning_rate * weight_deriv for weight, weight_deriv \
                          in zip(weights, weight_derivs) \
                      ]
        return biases, weights

if __name__ == '__main__':
    activation_func = sigmoid
    x = NeuralNetwork([2, 3, 4], activation_func)
    seed(10)
    input_length = x.num_nodes_per_layer[0]
    output_length = x.num_nodes_per_layer[-1]
    test_inputs = [randn(input_length), randn(input_length)]
    test_outputs = [map(activation_func, randn(output_length)), map(activation_func, randn(output_length))]
    test_biases = []
    test_weights = []
    for i, num_nodes in enumerate(x.num_nodes_per_layer[1:]):
        test_biases.append(randn(num_nodes))
        test_weights.append(randn(num_nodes, x.num_nodes_per_layer[i]))
    y = x.train(test_inputs, test_outputs)
    print y
    print 0