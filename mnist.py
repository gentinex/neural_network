import numpy as np
import cPickle as pickle
from neural_network import LearningMethod, NeuralNetwork

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
        vector_outputs = np.array([convert_to_mnist_vector(output) for output in outputs])
        data[1] = vector_outputs
    return new_data_sets[0], new_data_sets[1], new_data_sets[2]
    
''' exercise from neuralnetworksanddeeplearning.com: classify MNIST data set
    consisting of handwritten numbers '''
def mnist():
    training, validation, test = load_mnist()
    mnist_network = NeuralNetwork([784, 30, 10])
    # we tried L-BFGS-B here with 400 max iterations,
    # but it was inferior (slower + less accurate - 68% accuracy after one hour,
    # compared to 91% after 10 mins for SGD
    return mnist_network.train(training, \
                               validation, \
                               test, \
                               0.0002, \
                               5000, \
                               10, \
                               learning_method=LearningMethod('SGD', {'learning_rate' : 3.0}) \
                              )

if __name__ == '__main__':
    mnist()
