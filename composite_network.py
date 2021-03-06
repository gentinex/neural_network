import datetime
import numpy as np
from learning import LearningMethod, select_data
from scipy.optimize import fmin_l_bfgs_b

EPSILON = 1e-10

''' compose a neural network with a softmax classification layer.
    the last layer of the neural network may be combined with the neural network
    input to form the input to the softmax layer '''
class CompositeNetwork:
    def __init__(self, neural_network, softmax, concat=True):
        self.neural_network = neural_network
        self.softmax = softmax
        self.concat = concat
        
    def params(self):
        return self.neural_network.params(), self.softmax.weights
        
    def calc_softmax_inputs(self, inputs):
        activations, _, _ = self.neural_network.feedforward(inputs)
        if self.concat:
            return np.hstack((inputs, activations.T))
        else:
            return activations.T
        
    def feedforward(self, inputs):
        return self.softmax.probabilities(self.calc_softmax_inputs(inputs))
        
    def cost(self, inputs, outputs):
        activations = self.calc_softmax_inputs(inputs)
        return self.softmax.cost(activations, outputs)
        
    def backpropagate(self, inputs, outputs):
        softmax_inputs = self.calc_softmax_inputs(inputs)
        softmax_cost_deriv = self.softmax.backpropagate(softmax_inputs, outputs)
        probs = self.softmax.probabilities(softmax_inputs)
        neural_network_output_size = self.neural_network.num_nodes_per_layer[-1]
        used_softmax_weights = self.softmax.weights[:, -neural_network_output_size:]
        cost_pre_activation_deriv = \
            -(probs.T - outputs).dot(used_softmax_weights).T / float(len(inputs))
        neural_network_derivs = \
            self.neural_network.backpropagate(inputs, [], cost_pre_activation_deriv)
        return neural_network_derivs, softmax_cost_deriv
        
    ''' numerical derivative - for validating backpropagation '''
    def cost_deriv(self, inputs, outputs):
        neural_network_bias_derivs = \
            [np.zeros(bias.shape) for bias in self.neural_network.biases]
        neural_network_weight_derivs = \
            [np.zeros(weight.shape) for weight in self.neural_network.weights]
        softmax_weight_derivs = np.zeros(self.softmax.weights.shape)
        base_cost = self.cost(inputs, outputs)
        for i, bias in enumerate(self.neural_network.biases):
            for j, _ in enumerate(bias):
                self.neural_network.biases[i][j] += EPSILON
                neural_network_bias_derivs[i][j] = \
                    (self.cost(inputs, outputs) - base_cost) / EPSILON
                self.neural_network.biases[i][j] -= EPSILON
        for i, weight in enumerate(self.neural_network.weights):
            for j, weight_row in enumerate(weight):
                for k, _ in enumerate(weight_row):
                    self.neural_network.weights[i][j][k] += EPSILON
                    neural_network_weight_derivs[i][j][k] = \
                        (self.cost(inputs, outputs) - base_cost) / EPSILON
                    self.neural_network.weights[i][j][k] -= EPSILON
        for i, weight in enumerate(self.softmax.weights):
            for j, weight_row in enumerate(weight):
                self.softmax.weights[i][j] += EPSILON
                softmax_weight_derivs[i][j] = \
                    (self.cost(inputs, outputs) - base_cost) / EPSILON
                self.softmax.weights[i][j] -= EPSILON
        return (neural_network_bias_derivs, neural_network_weight_derivs), softmax_weight_derivs

    def unflatten_params(self, unrolled):
        num_total_params = len(unrolled)
        num_softmax_params = np.prod(self.softmax.weights.shape)
        num_neural_network_params = num_total_params - num_softmax_params
        unrolled_neural_network = unrolled[:num_neural_network_params]
        unrolled_softmax = unrolled[num_neural_network_params:]
        self.neural_network.unflatten_params(unrolled_neural_network)
        self.softmax.unflatten_params(unrolled_softmax)
        
    def flatten_params(self, (neural_network_params, softmax_params)):
        neural_network_flattened = \
            self.neural_network.flatten_params(neural_network_params)
        return np.concatenate((neural_network_flattened, softmax_params.ravel()))
        
    def cost_unrolled(self, unrolled, inputs, outputs):
        self.unflatten_params(unrolled)
        return self.cost(inputs, outputs)
        
    def cost_deriv_unrolled(self, unrolled, inputs, outputs):
        self.unflatten_params(unrolled)
        return self.flatten_params(self.backpropagate(inputs, outputs))
    
    def l_bfgs_b(self, inputs, outputs, max_iter):
        unrolled = self.flatten_params(self.params())
        bound_cost = lambda x: self.cost_unrolled(x, inputs, outputs)
        bound_cost_deriv = lambda x: self.cost_deriv_unrolled(x, inputs, outputs)
        optimal_unrolled, _, _ = fmin_l_bfgs_b(bound_cost, unrolled, bound_cost_deriv, maxiter=max_iter)
        return self.unflatten_params(optimal_unrolled)
    
    def evaluate(self, inputs, outputs):
        softmax_inputs = self.calc_softmax_inputs(inputs)
        return self.softmax.evaluate(softmax_inputs, outputs)
        
    def train(self, \
              training, \
              test, \
              batch_pct, \
              num_per_epoch, \
              num_epochs, \
              learning_method=LearningMethod('L-BFGS-B', {'max_iter' : 400}) \
             ):
        print 'Training started at', str(datetime.datetime.now())
        inputs, outputs = training
        for epoch in xrange(num_epochs):
            for _ in xrange(num_per_epoch):
                used_inputs, used_outputs = select_data(inputs, outputs, batch_pct)
                if learning_method.name == 'SGD':
                    self.gradient_descent(used_inputs, used_outputs, learning_method.paramsDict['learning_rate'])
                elif learning_method.name == 'L-BFGS-B':
                    self.l_bfgs_b(used_inputs, used_outputs, learning_method.paramsDict['max_iter'])
                else:
                    raise ValueError, 'SGD or L-BFGS-B are the only supported learning methods'
            pct_correct = self.evaluate(inputs, outputs) * 100.
            print 'Training epoch', str(epoch), ':', str(pct_correct), 'correct'
        print 'Training finished at', str(datetime.datetime.now())
        if test:
            pct_correct_test = self.evaluate(test[0], test[1]) * 100.
            print 'Test:', str(pct_correct_test), 'correct'
