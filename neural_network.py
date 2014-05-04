# TODO:
# -get visualizations for the ones the network does wrong
# -add regularization and see how it performs
# -go back to ufldl
# -set up autoencoder
# -do exercise from ufldl tutorial
# -learn about svm approach

import cPickle as pickle
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import numpy.random as random
import sys
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
        # learning rate is quite important - note e.g. that linear seems to
        # require much smaller rates than sigmoid to properly converge
        self.learning_rate = 0.3
        random.seed(1)
        self.biases = []
        self.weights = []
        for i, num_nodes in enumerate(self.num_nodes_per_layer[1:]):
            self.biases.append(random.randn(num_nodes))
            self.weights.append(random.randn(num_nodes, self.num_nodes_per_layer[i]))
        
    def check_input(self, input):
        assert len(input) == self.num_nodes_per_layer[0]

    def feedforward(self, input):
        self.check_input(input)
        vectorized_activation = np.vectorize(self.activation_func.eval)
        activations = []
        activation = input
        activations.append(activation)
        pre_activations = []
        for bias, weight in zip(self.biases, self.weights):
            with_one = np.insert(activation, 0, 1.)
            bias_and_weight = np.hstack((np.array([bias]).T, weight))
            pre_activation = bias_and_weight.dot(with_one)
            pre_activations.append(pre_activation)
            activation = vectorized_activation(pre_activation)
            activations.append(activation)
        return activation, activations, pre_activations
        
    def backpropagate(self, inputs, outputs):
        bias_derivs = [np.zeros(bias.shape) for bias in self.biases]
        weight_derivs = [np.zeros(weight.shape) for weight in self.weights]
        for i, (input, output) in enumerate(zip(inputs, outputs)):
            activation, activations, pre_activations = \
                self.feedforward(input)
            vectorized_activation_deriv = np.vectorize(self.activation_func.deriv)
            pre_activation_derivs = \
                [vectorized_activation_deriv(pre_activation) \
                     for pre_activation in pre_activations \
                ]
            cost_pre_activation_deriv = \
                (activation - output) * pre_activation_derivs[-1]
            cost_pre_activation_derivs = [cost_pre_activation_deriv]
            for pre_activation_deriv, weight in \
                reversed(zip(pre_activation_derivs[:-1], self.weights[1:])):
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
        
    def cost(self, inputs, outputs):
        net_cost = 0
        for input, output in zip(inputs, outputs):
            predicted_output, _, _ = self.feedforward(input, self.biases, self.weights)
            net_cost += sum(0.5 * (predicted_output - output) ** 2.)
        return net_cost
        
    def cost_deriv(self, inputs, outputs):
        bias_derivs = [np.zeros(bias.shape) for bias in self.biases]
        weight_derivs = [np.zeros(weight.shape) for weight in self.weights]
        base_cost = self.cost(inputs, outputs)
        for i, bias in enumerate(self.biases):
            for j, bias_elt in enumerate(bias):
                self.biases[i][j] += EPSILON
                bias_derivs[i][j] = \
                    (self.cost(inputs, outputs) - base_cost) / EPSILON
                self.biases[i][j] -= EPSILON
        for i, weight in enumerate(self.weights):
            for j, weight_row in enumerate(weight):
                for k, weight_col in enumerate(weight_row):
                    self.weights[i][j][k] += EPSILON
                    weight_derivs[i][j][k] = \
                        (self.cost(inputs, outputs) - base_cost) / EPSILON
                    self.weights[i][j][k] -= EPSILON
        return bias_derivs, weight_derivs

    def gradient_descent(self, inputs, outputs, batch_pct = 1.):
        assert batch_pct >= 0. and batch_pct <= 1.
        num_inputs = len(inputs)
        num_selected = max(1, int(num_inputs * batch_pct))
        all_indices = xrange(num_inputs)
        if batch_pct == 1.:
            used_indices = np.array(all_indices)
        else:
            used_indices = random.choice(all_indices, num_selected, False)
        used_inputs = [inputs[index] for index in used_indices]
        used_outputs = [outputs[index] for index in used_indices]
        bias_derivs, weight_derivs = \
            self.backpropagate(used_inputs, used_outputs)
        self.biases = \
            [bias - self.learning_rate * bias_deriv \
                 for bias, bias_deriv in zip(self.biases, bias_derivs) \
            ]
        self.weights = \
            [weight - self.learning_rate * weight_deriv \
                for weight, weight_deriv in zip(self.weights, weight_derivs) \
            ]

    def evaluate(self, inputs, outputs, show_errors=False):
        predicted_output = \
            [np.argmax(self.feedforward(input)[0]) for input in inputs]
        actual_output = [np.argmax(output) for output in outputs]
        if show_errors:
            for a, b, c in zip(inputs, predicted_output, actual_output):
                if b != c:
                    imgplot = plt.imshow(np.reshape(a, (28,28)))
                    imgplot.set_cmap('binary')
                    plt.title('Predicted ' + str(b) + ', actual is ' + str(c))
                    plt.show()
        comparison = [a == b for a, b in zip(predicted_output, actual_output)]
        num_correct = len(list(x for x in comparison if x))
        return float(num_correct) / float(len(inputs))
            
    def train(self, training, validation, test, batch_pct, num_per_epoch, num_epochs):
        print 'Started at', str(datetime.datetime.now())
        inputs, outputs = training
        for epoch in xrange(num_epochs):
            for run in xrange(num_per_epoch):
                self.gradient_descent(inputs, outputs, batch_pct)
            pct_correct = self.evaluate(inputs, outputs) * 100.
            print 'Epoch', str(epoch), ':', str(pct_correct), 'correct'
        vinputs, voutputs = validation
        pct_correct_validation = self.evaluate(vinputs, voutputs, True) * 100.
        print 'Validation:', str(epoch), ':', str(pct_correct_validation), 'correct'
        tinputs, toutputs = test
        pct_correct_test = self.evaluate(tinputs, toutputs) * 100.
        print 'Test:', str(epoch), ':', str(pct_correct_test), 'correct'
        print 'Finished at', str(datetime.datetime.now())

def load_mnist():
    def convert_to_mnist_vector(output):
        vector_output = np.zeros(10)
        vector_output[output] = 1.0
        return vector_output
    with open('../../data/mnist/mnist.pkl', 'rb') as mnist_pkl:
        training, validation, test = pickle.load(mnist_pkl)
    new_data_sets = [[data[0], data[1]] for data in [training, validation, test]]
    for data in new_data_sets:
        outputs = data[1]
        vector_outputs = [convert_to_mnist_vector(output) for output in outputs]
        data[1] = vector_outputs
    return new_data_sets[0], new_data_sets[1], new_data_sets[2]

def sample_test():
    activation_func = sigmoid
    x = NeuralNetwork([2, 3, 4], activation_func)
    seed(10)
    input_length = x.num_nodes_per_layer[0]
    output_length = x.num_nodes_per_layer[-1]
    test_inputs = [random.randn(input_length), random.randn(input_length)]
    test_outputs = [map(activation_func, random.randn(output_length)), \
                    map(activation_func, random.randn(output_length)) \
                   ]
    test_biases = []
    test_weights = []
    for i, num_nodes in enumerate(x.num_nodes_per_layer[1:]):
        test_biases.append(random.randn(num_nodes))
        test_weights.append(random.randn(num_nodes, x.num_nodes_per_layer[i]))
    return x.train((test_inputs, test_outputs), 1., 100, 1)
    
def sample_linear_test():
    # some things to note:
    # -it does seem like the algorithm works correctly
    # -univariate converges relatively quickly (10k passes is more than enough)
    # -multivariate is slower (for z=a+bx+cy, need 100k passes to get sort of close)
    activation_func = lambda x: x
    x = NeuralNetwork([2, 1], activation_func)
    y = x.train(([[7., 2.], [8., 1.], [4., 3.], [2., 5.]], [[4.], [2.], [2.], [1.]]), 1., 1000, 100)
    print x.biases, x.weights
    
def mnist_test():
    training, validation, test = load_mnist()
    nn = NeuralNetwork([784, 30, 10])
    return nn.train(training, validation, test, 0.0002, 500, 10)
    
if __name__ == '__main__':
    mnist_test()
