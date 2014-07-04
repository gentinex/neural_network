import numpy as np
import numpy.random as random
from composite_network import CompositeNetwork
from learning import LearningMethod
from mnist import load_mnist
from neural_network import NeuralNetwork, SparsityParams
from scipy.optimize import fmin_l_bfgs_b
from softmax import Softmax

# accuracy comments:
# -without fine-tuning: 95.3% training, 94.6% test
# -with fine-tuning: 

def toy_example():
    random.seed(1)
    autoencoder_network = NeuralNetwork([4, 3, 5])
    softmax = Softmax(2, 5, regularization=0.01)
    composite_network = CompositeNetwork(autoencoder_network, softmax, False)
    inputs = random.randn(5, 4)
    outputs = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.], [1., 0.]])
    print composite_network.backpropagate(inputs, outputs)
    print composite_network.cost_deriv(inputs, outputs)

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
    print 'Training first autoencoder...'
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
    print 'Training second autoencoder...'
    autoencoder1.train([encoded0.T, encoded0.T],
                       [], \
                       1., \
                       1, \
                       1, \
                       learning_method=LearningMethod('L-BFGS-B', {'max_iter' : 400}) \
                      )
    encoded1, _, _ = autoencoder1.feedforward(encoded0.T, 1)
    
    encoded_test0, _, _ = autoencoder0.feedforward(test_inputs, 1)
    encoded_test1, _, _ = autoencoder1.feedforward(encoded_test0.T, 1)
    
    concat = False
    if concat:
        used_test_inputs = np.hstack((test_inputs, encoded_test1.T))
        used_train_inputs = np.hstack((inputs, encoded1.T))
        softmax_input_size = 196 + 784
    else:
        used_test_inputs = encoded_test1.T
        used_train_inputs = encoded1.T
        softmax_input_size = 196

    softmax = Softmax(10, softmax_input_size, regularization=1e-4)
    print 'Training softmax classifier...'
    softmax.train([used_train_inputs, outputs], \
                  [used_test_inputs, test_outputs], \
                  1., \
                  1, \
                  1, \
                  LearningMethod('L-BFGS-B', {'max_iter' : 100})
                 )
                 
    # no regularization or sparsity for autoencoder levels in composite network (why?)
    autoencoder_network = \
        NeuralNetwork([784, 196, 196], \
                      #sparsity_params=SparsityParams(0.1, 3.) \
                      #regularization=1e-4 \
                     )
    autoencoder_network.biases[0] = autoencoder0.biases[0]
    autoencoder_network.biases[1] = autoencoder1.biases[0]
    autoencoder_network.weights[0] = autoencoder0.weights[0]
    autoencoder_network.weights[1] = autoencoder1.weights[0]
    
    composite_network = CompositeNetwork(autoencoder_network, softmax, concat)
    print 'Fine-tuning...'
    composite_network.train([inputs, outputs], \
                            [test_inputs, test_outputs], \
                            1., \
                            1, \
                            1, \
                            LearningMethod('L-BFGS-B', {'max_iter' : 400})
                           )
                 
if __name__ == '__main__':
    toy_example()
#    stacked_autoencoder_mnist()
