import numpy as np
import numpy.random as random
from learning import LearningMethod
from mnist import load_mnist
from neural_network import NeuralNetwork, SparsityParams
from softmax import Softmax

def stacked_autoencoder_mnist():
    training, test = load_mnist()
    inputs, outputs = training
    test_inputs, test_outputs = test
    random.seed(1)
    autoencoder0 = \
        NeuralNetwork([784, 196, 784], \
                      regularization=0.003, \
                      sparsity_params=SparsityParams(0.1, 3.) \
                     )
    autoencoder0.train([inputs, inputs],
                       [], \
                       1., \
                       1, \
                       1, \
                       learning_method=LearningMethod('L-BFGS-B', {'max_iter' : 400}) \
                      )
    encoded0, _, _ = autoencoder0.feedforward(inputs, 1)

    autoencoder1 = \
        NeuralNetwork([196, 196, 196], \
                      regularization=0.003, \
                      sparsity_params=SparsityParams(0.1, 3.) \
                     )
    autoencoder1.train([encoded0.T, encoded0.T],
                       [], \
                       1., \
                       1, \
                       1, \
                       learning_method=LearningMethod('L-BFGS-B', {'max_iter' : 400}) \
                      )
    encoded1, _, _ = autoencoder1.feedforward(encoded0.T, 1)
    
    concat_train_inputs = np.hstack((inputs, encoded1.T))
    softmax = Softmax(10, 196 + 784, regularization=1e-4)
    softmax.train([concat_train_inputs, outputs], \
                  [], \
                  1., \
                  1, \
                  1, \
                  LearningMethod('L-BFGS-B', {'max_iter' : 100})
                 )

    encoded_test0, _, _ = autoencoder0.feedforward(test_inputs, 1)
    encoded_test1, _, _ = autoencoder1.feedforward(encoded_test0.T, 1)
    concat_test_inputs = np.hstack((test_inputs, encoded_test1.T))
    test_correct = softmax.evaluate(concat_test_inputs, test_outputs) * 100.
    print 'Test set accuracy after softmax train:', str(test_correct)
                 
    autoencoder_network = \
        NeuralNetwork([784, 196, 196], \
                      regularization=0.003, \
                      sparsity_params=SparsityParams(0.1, 3.) \
                     )
    autoencoder_network.biases[0] = autoencoder0.biases[0]
    autoencoder_network.biases[1] = autoencoder1.biases[0]
    autoencoder_network.weights[0] = autoencoder0.weights[0]
    autoencoder_network.weights[1] = autoencoder1.weights[0]
                 
if __name__ == '__main__':
    stacked_autoencoder_mnist()
