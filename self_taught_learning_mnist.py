import numpy as np
import numpy.random as random
from learning import LearningMethod
from mnist import load_mnist
from neural_network import NeuralNetwork, SparsityParams
from softmax import Softmax

# accuracy notes (but do note that the supervised training / test are only on
# a subset of mnist):
# -train autoencoder, train softmax on feedforward only: 98.7% train, 98.5% test
# -train autoencoder, train softmax on feedforward + orig: 99.6% train, 98.5% test
# -no autoencoder, train softmax on orig: 98.3% train, 97.0% test

def self_taught_learning_mnist():
    training, _ = load_mnist()
    full_data = zip(training[0], training[1])
    unlabeled = np.array([input for input, output in full_data \
                          if sum(output[5:]) == 1.])
    random.seed(1)
    autoencoder_mnist_network = \
        NeuralNetwork([784, 196, 784], \
                      regularization=0.003, \
                      sparsity_params=SparsityParams(0.1, 3.) \
                     )
    print 'Training unlabeled...'
    autoencoder_mnist_network.train([unlabeled, unlabeled], \
                                    [], \
                                    1., \
                                    1, \
                                    1, \
                                    learning_method=LearningMethod('L-BFGS-B', {'max_iter' : 400}) \
                                   )

    labeled = [(input, output) for input, output in full_data \
               if sum(output[0:5]) == 1.]
    labeled_train = labeled[0:(len(labeled) / 2)]
    labeled_test = labeled[(len(labeled) / 2):]
    labeled_train_inputs_list, labeled_train_outputs_list = zip(*labeled_train)
    labeled_train_inputs = np.array(labeled_train_inputs_list)
    labeled_train_outputs = np.array(labeled_train_outputs_list)
    feedforward_train_inputs, _, _ = \
        autoencoder_mnist_network.feedforward(labeled_train_inputs, 1)

    concat = True
    if concat:
        softmax_train_inputs = np.hstack((labeled_train_inputs, feedforward_train_inputs.T))
        softmax_input_size = 196 + 784
    else:
        softmax_train_inputs = feedforward_train_inputs.T
        softmax_input_size = 196
    softmax = Softmax(10, softmax_input_size, regularization = 1e-4)
    print 'Training labeled...'
    softmax.train([softmax_train_inputs, labeled_train_outputs], \
                  [], \
                  1., \
                  1, \
                  1, \
                  LearningMethod('L-BFGS-B', {'max_iter' : 100})
                 )
    labeled_test_inputs_list, labeled_test_outputs_list = zip(*labeled_test)
    labeled_test_inputs = np.array(labeled_test_inputs_list)
    labeled_test_outputs = np.array(labeled_test_outputs_list)
    feedforward_test_inputs, _, _ = \
        autoencoder_mnist_network.feedforward(labeled_test_inputs, 1)
    if concat:
        softmax_test_inputs = np.hstack((labeled_test_inputs, feedforward_test_inputs.T))
    else:
        softmax_test_inputs = feedforward_test_inputs.T
    test_correct = softmax.evaluate(softmax_test_inputs, labeled_test_outputs) * 100.
    print 'Test set accuracy:', str(test_correct)

if __name__ == '__main__':
    self_taught_learning_mnist()
