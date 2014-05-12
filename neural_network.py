# TODO:
# -do sparsity exercise from ufldl tutorial
# -why can we not replicate what's done in ufldl:
#  -does it matter if we don't normalize our image inputs? I thought they were
#   always between 0 and 1?
#  -did we implement sparsity correctly? i.e., wouldn't we need to cycle through
#   all training examples (or the present subset) before calcing sparsity penalty? 
# -why do we keep getting errors with full gradient descent rather than stochastic?
#  probably has to do with learning rate magnitude relative to batch size..
# -set up better vectorization
# -profile
# -learn about svm approach to mnist

import copy
import cPickle as pickle
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import numpy.random as random
import scipy.io
import sys
from scipy.misc import derivative

EPSILON = 1e-10

def sigmoid(x):
    return 1. / (1. + math.exp(-x))

def display_image(image, title=''):
    imgplot = plt.imshow(image)
    imgplot.set_cmap('binary')
    plt.title(title)
    plt.show()

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
    def __init__(self, \
                 num_nodes_per_layer, \
                 activation_func=sigmoid, \
                 regularization=0., \
                 sparsity=None
                ):
        self.num_nodes_per_layer = num_nodes_per_layer
        self.activation_func = ActivationFunction(activation_func)
        
        # clearly a sweet spot for regularization..for MNIST, 
        # 0.001 worked great, 0.01 and 0.0001 not so much
        self.regularization = regularization
        
        # learning rate is quite important - note e.g. that linear seems to
        # require much smaller rates than sigmoid to properly converge
        self.learning_rate = 0.3
        
        self.sparsity = sparsity
        self.sparsity_weight = 0.1
        
        random.seed(1)
        self.biases = []
        self.weights = []
        for i, num_nodes in enumerate(self.num_nodes_per_layer[1:]):
            self.biases.append(random.randn(num_nodes))
            self.weights.append(random.randn(num_nodes, self.num_nodes_per_layer[i]))
            
    ''' check that input is valid '''
    def check_input(self, input):
        assert len(input) == self.num_nodes_per_layer[0]

    ''' feed input forward through the network '''
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

    ''' fast calculation of network derivatives with respect to weights / biases '''
    def backpropagate(self, inputs, outputs):
        bias_derivs = [np.zeros(bias.shape) for bias in self.biases]
        weight_derivs = [self.regularization * copy.deepcopy(weight) \
                         for weight in self.weights \
                        ]
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
            for pre_activation_deriv, weight, activation_per_layer in \
                reversed(zip(pre_activation_derivs[:-1], self.weights[1:], activations[1:-1])):
                if self.sparsity != None:
                    avg_activation = np.mean(activation_per_layer)
                    sparsity_cost = \
                        self.sparsity_weight \
                        * (-self.sparsity / avg_activation \
                           + (1 - self.sparsity) / (1 - avg_activation) \
                          )
                else:
                    sparsity_cost = 0.
                cost_pre_activation_deriv = \
                    (weight.T.dot(cost_pre_activation_deriv) + sparsity_cost) \
                    * pre_activation_deriv
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

    ''' standard cost function - used for validating backpropagation '''
    def cost(self, inputs, outputs):
        net_cost = 0
        for input, output in zip(inputs, outputs):
            predicted_output, _, _ = self.feedforward(input, self.biases, self.weights)
            net_cost += sum(0.5 * (predicted_output - output) ** 2.)
        return net_cost
        
    ''' derivative of standard cost function - used for validating backpropagation '''
    def cost_deriv(self, inputs, outputs):
        bias_derivs = [np.zeros(bias.shape) for bias in self.biases]
        weight_derivs = [self.regularization * copy.deepcopy(weight) \
                         for weight in self.weights \
                        ]
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

    ''' gradient_descent, to find optimal weights / biases '''
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

    ''' evaluate the network performance '''
    def evaluate(self, inputs, outputs, show_errors=False):
        predicted_output = \
            [np.argmax(self.feedforward(input)[0]) for input in inputs]
        actual_output = [np.argmax(output) for output in outputs]
        if show_errors:
            for a, b, c in zip(inputs, predicted_output, actual_output):
                if b != c:
                    display_image(np.reshape(a, (28, 28)), \
                                  'Predicted ' + str(b) + ', actual is ' + str(c)
                                 )
        comparison = [a == b for a, b in zip(predicted_output, actual_output)]
        num_correct = len(list(x for x in comparison if x))
        return float(num_correct) / float(len(inputs))

    ''' calibrate weights and biases of the network with supervised learning and
        stochastic gradient descent. at the end of each epoch, we run the network
        on the full training set to determine training accuracy.
    '''
    def train(self, training, validation, test, batch_pct, num_per_epoch, num_epochs):
        print 'Started at', str(datetime.datetime.now())
        inputs, outputs = training
        for epoch in xrange(num_epochs):
            for run in xrange(num_per_epoch):
                self.gradient_descent(inputs, outputs, batch_pct)
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
    y = x.train(([[7., 2.], [8., 1.], [4., 3.], [2., 5.]], [[4.], [2.], [2.], [1.]]), 1., 1000, 100)
    print x.biases, x.weights
    
def mnist_test():
    training, validation, test = load_mnist()
    mnist_network = NeuralNetwork([784, 30, 10], regularization=0.001)
    return mnist_network.train(training, validation, test, 0.0002, 500, 10)

def generate_random_image_slice(images, height, width):
    height_images, width_images, num_images = images.shape
    image_index = random.randint(0, num_images)
    image_height = random.randint(0, height_images - height)
    image_width = random.randint(0, width_images - width)
    return np.ndarray.flatten(images[image_height:(image_height + height), \
                                     image_width:(image_width + width), \
                                     image_index \
                                    ] \
                             )
    
def sparse_autoencoder_test():
    images = scipy.io.loadmat('../../data/SparseAutoEncoder/IMAGES.mat')['IMAGES']
    random.seed(100)
    image_slices = np.array([generate_random_image_slice(images, 8, 8) for i in xrange(10000)])
    autoencoder_network = NeuralNetwork([64, 25, 64], regularization=0.001, sparsity=0.5)
    autoencoder_network.train([image_slices, image_slices], [], [], 0.001, 10000, 10)
    calibrated_weights = autoencoder_network.weights
    final_image = np.zeros((40, 40))
    xmin, ymin, xmax, ymax = 0, 0, 8, 8
    for wt in calibrated_weights[0]:
        reshaped = np.reshape(wt, (8, 8))
        final_image[xmin:xmax, ymin:ymax] = reshaped
        if xmax == 40:
            xmin, xmax = 0, 8
            ymin = ymin + 8
            ymax = ymax + 8
        else:
            xmin = xmin + 8
            xmax = xmax + 8
    display_image(final_image)
    
if __name__ == '__main__':
    #mnist_test()
    sparse_autoencoder_test()
