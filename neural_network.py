# TODO:
# -[low priority] find out why, when we normalize to [0, 1] rather than [0.1, 0.9]
#  in sparse_autoencoder, we seem to get bad results (though this isn't the case
#  for the MATLAB implementation)
# -how does autoencoder compare to PCA, as a way to determine essential features?
# -also see how bfgs does on mnist
# -put in pre-commit hook to run numerical gradient check on simple example
# -profile (maybe look into gpus??)
# -learn about svm approach to mnist

import copy
import datetime
import itertools
import math
import numpy as np
import numpy.random as random
from images import display_image
from scipy.misc import derivative
from scipy.optimize import fmin_l_bfgs_b

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

class SparsityParams:
    def __init__(self, sparsity, weight):
        self.sparsity = sparsity
        self.weight = weight

class LearningMethod:
    def __init__(self, name, paramsDict={}):
        self.name = name
        self.paramsDict = paramsDict

class NeuralNetwork:
    def __init__(self, \
                 num_nodes_per_layer, \
                 activation_func=sigmoid, \
                 regularization=0., \
                 sparsity_params=None
                ):
        assert all(num_nodes > 0 for num_nodes in num_nodes_per_layer)
        self.num_nodes_per_layer = num_nodes_per_layer
        self.activation_func = ActivationFunction(activation_func)

        self.regularization = regularization
        self.sparsity_params = sparsity_params

        random.seed(1)
        self.biases = []
        self.weights = []
        for i, num_nodes in enumerate(self.num_nodes_per_layer[1:]):
            self.biases.append(random.randn(num_nodes))
            self.weights.append(random.randn(num_nodes, self.num_nodes_per_layer[i]))

    ''' check that inputs are valid '''
    def check_inputs(self, inputs):
        assert len(inputs) > 0 and inputs.shape[1] == self.num_nodes_per_layer[0]

    ''' feed input forward through the network '''
    def feedforward(self, inputs):
        self.check_inputs(inputs)
        vectorized_activation = np.vectorize(self.activation_func.eval)
        activations = []
        activation = inputs.T
        activations.append(activation)
        pre_activations = []
        for bias, weight in zip(self.biases, self.weights):
            with_one = np.insert(activation, 0, 1., axis=0)
            bias_and_weight = np.insert(weight, 0, bias, axis=1)
            pre_activation = bias_and_weight.dot(with_one)
            pre_activations.append(pre_activation)
            activation = vectorized_activation(pre_activation)
            activations.append(activation)
        return activation, activations, pre_activations

    ''' because sparsity calculation requires first pass through, do this as
        a separate calc '''
    def do_first_pass(self, inputs):
        first_pass = self.feedforward(inputs)
        _, activations, _ = first_pass
        inner_activations = activations[1:-1]
        avg_activations = [[]] * len(inner_activations)
        if self.sparsity_params != None:
            avg_activations = [np.mean(inner_activation, 1) \
                               for inner_activation in inner_activations \
                              ]
        else:
            avg_activations = [np.zeros(inner_activation.shape[0]) \
                               for inner_activation in inner_activations
                              ]
        return first_pass, avg_activations

    ''' fast calculation of network derivatives with respect to weights / biases '''
    def backpropagate(self, used_inputs, used_outputs):
        bias_derivs = [np.zeros(bias.shape) for bias in self.biases]
        # regularization derivative
        weight_derivs = [self.regularization * copy.deepcopy(weight) \
                         for weight in self.weights \
                        ]
        data_size = float(len(used_inputs))
        (activation, activations, pre_activations), avg_activations = \
            self.do_first_pass(used_inputs)
        vectorized_activation_deriv = np.vectorize(self.activation_func.deriv)
        pre_activation_derivs = \
            [vectorized_activation_deriv(pre_activation) \
                 for pre_activation in pre_activations \
            ]
        cost_pre_activation_deriv = \
            (activation - used_outputs.T) * pre_activation_derivs[-1] / data_size
        cost_pre_activation_derivs = [cost_pre_activation_deriv]

        for pre_activation_deriv, weight, avg_activation in \
            reversed(zip(pre_activation_derivs[:-1], self.weights[1:], avg_activations)):
            # sparsity derivative
            if self.sparsity_params != None:
                sparsity_cost_deriv = \
                    self.sparsity_params.weight \
                    * (-self.sparsity_params.sparsity / avg_activation \
                       + (1 - self.sparsity_params.sparsity) / (1 - avg_activation) \
                      ) / data_size
            else:
                sparsity_cost_deriv = np.zeros(avg_activation.shape)
            # standard derivative
            cost_pre_activation_deriv = \
                (weight.T.dot(cost_pre_activation_deriv) + sparsity_cost_deriv.reshape(-1, 1)) \
                * pre_activation_deriv
            cost_pre_activation_derivs.insert(0, cost_pre_activation_deriv)

        bias_derivs = \
            [np.sum(cost_pre_activation_deriv, 1) \
             for cost_pre_activation_deriv in cost_pre_activation_derivs \
            ]
        weight_derivs = \
            [self.regularization * weight \
             + cost_pre_activation_deriv.dot(activation_b.T) \
             for weight, cost_pre_activation_deriv, activation_b \
                 in zip(self.weights, cost_pre_activation_derivs, activations[:-1])\
            ]
        return bias_derivs, weight_derivs

    ''' standard cost function - used for validating backpropagation '''
    def cost(self, inputs, outputs):
        data_size = float(len(inputs))
        (activation, _, _), avg_activations = self.do_first_pass(inputs)
        # regularization cost
        net_cost = (self.regularization / 2.) * sum(np.sum(weight ** 2.) for weight in self.weights)
        # standard cost
        net_cost += np.sum((activation - outputs.T) ** 2.) * 0.5 / data_size
        # sparsity cost
        if self.sparsity_params != None:
            for avg_activation in avg_activations:
                base_sparsity_penalty = \
                    self.sparsity_params.sparsity * np.log(self.sparsity_params.sparsity / avg_activation) \
                    + (1 - self.sparsity_params.sparsity)  * np.log((1 - self.sparsity_params.sparsity) / (1 - avg_activation))
                net_cost += self.sparsity_params.weight * sum(base_sparsity_penalty)
        return net_cost
        
    ''' numerical derivative of standard cost function - used for validating backpropagation '''
    def cost_deriv(self, inputs, outputs):
        bias_derivs = [np.zeros(bias.shape) for bias in self.biases]
        weight_derivs = [self.regularization * copy.deepcopy(weight) \
                         for weight in self.weights \
                        ]
        base_cost = self.cost(inputs, outputs)
        for i, bias in enumerate(self.biases):
            for j, _ in enumerate(bias):
                self.biases[i][j] += EPSILON
                bias_derivs[i][j] = \
                    (self.cost(inputs, outputs) - base_cost) / EPSILON
                self.biases[i][j] -= EPSILON
        for i, weight in enumerate(self.weights):
            for j, weight_row in enumerate(weight):
                for k, _ in enumerate(weight_row):
                    self.weights[i][j][k] += EPSILON
                    weight_derivs[i][j][k] = \
                        (self.cost(inputs, outputs) - base_cost) / EPSILON
                    self.weights[i][j][k] -= EPSILON
        return bias_derivs, weight_derivs
        
    ''' flatten weights and biases '''
    def flatten_params(self, weight_list, bias_list):
        unrolled_weights = list(itertools.chain(*[weight.flatten() for weight in weight_list]))
        unrolled_biases = list(itertools.chain(*bias_list))
        return np.array(unrolled_weights + unrolled_biases)

    ''' unflatten weights and biases '''
    def unflatten_params(self, unrolled, column_major=False):
        start_index = 0
        for i, weight in enumerate(self.weights):
            weight_shape = weight.shape
            weight_len = weight_shape[0] * weight_shape[1]
            order = 'C'
            if column_major:
                order = 'F'
            self.weights[i] = np.reshape(unrolled[start_index:(start_index + weight_len)], weight_shape, order)
            start_index = start_index + weight_len
        for i, bias in enumerate(self.biases):
            bias_len = len(bias)
            self.biases[i] = unrolled[start_index:(start_index + bias_len)]
            start_index = start_index + bias_len

    ''' cost function over a vector '''
    def cost_unrolled(self, unrolled, used_inputs, used_outputs):
        self.unflatten_params(unrolled)
        return self.cost(used_inputs, used_outputs)

    ''' cost derivative over a vector '''
    def cost_deriv_unrolled(self, unrolled, used_inputs, used_outputs):
        self.unflatten_params(unrolled)
        bias_derivs, weight_derivs = self.backpropagate(used_inputs, used_outputs)
        return self.flatten_params(weight_derivs, bias_derivs)
    
    ''' select a subset of the data for training '''
    def select_data(self, inputs, outputs, batch_pct=1.):
        assert batch_pct >= 0. and batch_pct <= 1.
        if batch_pct == 1.:
            used_inputs = inputs
            used_outputs = outputs
        else:
            num_inputs = len(inputs)
            num_selected = max(1, int(num_inputs * batch_pct))
            all_indices = xrange(num_inputs)
            random_indices = tuple(random.choice(all_indices, num_selected, False))
            used_inputs = inputs[random_indices, ...]
            used_outputs = outputs[random_indices, ...]
        return used_inputs, used_outputs

    ''' gradient_descent, to find optimal weights / biases '''
    def gradient_descent(self, used_inputs, used_outputs, learning_rate):
        bias_derivs, weight_derivs = \
            self.backpropagate(used_inputs, used_outputs)
        self.biases = \
            [bias - learning_rate * bias_deriv \
                 for bias, bias_deriv in zip(self.biases, bias_derivs) \
            ]
        self.weights = \
            [weight - learning_rate * weight_deriv \
                for weight, weight_deriv in zip(self.weights, weight_derivs) \
            ]
    
    def l_bfgs_b(self, used_inputs, used_outputs, maxIter):
        unrolled = self.flatten_params(self.weights, self.biases)
        bound_cost = lambda x: self.cost_unrolled(x, used_inputs, used_outputs)
        bound_cost_deriv = lambda x: self.cost_deriv_unrolled(x, used_inputs, used_outputs)
        optimal_unrolled, _, _ = fmin_l_bfgs_b(bound_cost, unrolled, bound_cost_deriv, maxiter=maxIter)
        self.unflatten_params(optimal_unrolled)
            
    ''' evaluate the network performance '''
    def evaluate(self, inputs, outputs, show_errors=False):
        predicted_outputs = self.feedforward(inputs)[0].argmax(0)
        actual_outputs = outputs.argmax(1)
        if show_errors:
            for a, b, c in zip(inputs, predicted_outputs, actual_outputs):
                if b != c:
                    display_image(np.reshape(a, (28, 28)), \
                                  'Predicted ' + str(b) + ', actual is ' + str(c)
                                 )
        comparison = [a == b for a, b in zip(predicted_outputs, actual_outputs)]
        num_correct = len(list(x for x in comparison if x))
        return float(num_correct) / float(len(inputs))

    ''' calibrate weights and biases of the network with supervised learning and
        stochastic gradient descent. at the end of each epoch, we run the network
        on the full training set to determine training accuracy.
    '''
    def train(self, \
              training, \
              validation, \
              test, \
              batch_pct, \
              num_per_epoch, \
              num_epochs, \
              learning_method=LearningMethod('SGD', {'learning_rate' : 0.1}) \
             ):
        print 'Training started at', str(datetime.datetime.now())
        inputs, outputs = training
        for epoch in xrange(num_epochs):
            for _ in xrange(num_per_epoch):
                used_inputs, used_outputs = self.select_data(inputs, outputs, batch_pct)
                if learning_method.name == 'SGD':
                    self.gradient_descent(used_inputs, used_outputs, learning_method.paramsDict['learning_rate'])
                elif learning_method.name == 'L-BFGS-B':
                    self.l_bfgs_b(used_inputs, used_outputs, learning_method.paramsDict['max_iter'])
                else:
                    raise ValueError, 'SGD or L-BFGS-B are the only supported learning methods'
            pct_correct = self.evaluate(inputs, outputs) * 100.
            print 'Epoch', str(epoch), ':', str(pct_correct), 'correct'
        if validation:
            vinputs, voutputs = validation
            pct_correct_validation = self.evaluate(vinputs, voutputs) * 100.
            print 'Validation:', str(epoch), ':', str(pct_correct_validation), 'correct'
        if test:
            tinputs, toutputs = test
            pct_correct_test = self.evaluate(tinputs, toutputs) * 100.
            print 'Test:', str(epoch), ':', str(pct_correct_test), 'correct'
        print 'Training finished at', str(datetime.datetime.now())

def sample_test():
    activation_func = sigmoid
    x = NeuralNetwork([2, 3, 4], activation_func)
    random.seed(10)
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
    _ = x.train(([[7., 2.], [8., 1.], [4., 3.], [2., 5.]], [[4.], [2.], [2.], [1.]]), 1., 1000, 100)
    print x.biases, x.weights
    
if __name__ == '__main__':
    sample_linear_test()
