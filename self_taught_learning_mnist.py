import numpy as np
from learning import LearningMethod
from mnist import load_mnist
from neural_network import NeuralNetwork, SparsityParams
from softmax import Softmax

def self_taught_learning_mnist():
    training, test = load_mnist()
    full_inputs = np.vstack((training[0], test[0]))
    full_outputs = np.vstack((training[1], test[1]))
    unlabeled = [input for input, output in zip(full_inputs, full_outputs) \
                 if sum(output[5:]) == 1.]
    print len(full_inputs), len(unlabeled)
    # random.seed(1)
    # autoencoder_mnist_network = \
        # NeuralNetwork([784, 196, 784], \
                      # regularization=0.003, \
                      # sparsity_params=SparsityParams(0.1, 3.) \
                     # )
    # autoencoder_mnist_network.train([autoencoder_data, autoencoder_data], \
                                    # [], \
                                    # 1., \
                                    # 1, \
                                    # 1, \
                                    # learning_method=LearningMethod('L-BFGS-B', {'max_iter' : 400}) \
                                   # )

    # labeled = [input, output for input, output in full_data if sum(output[0:5]) == 1.]
    # labeled_train = labeled[0:(len(labeled) / 2)]
    # labeled_test = labeled[(len(labeled) / 2):]
    # labeled_train_inputs, labeled_train_outputs = *labeled_train
    # softmax_training_inputs, _, _ = \
        # autoencoder_mnist_network.feedforward(labeled_train_inputs, 1)
    # softmax_training = zip(softmax_training_inputs, labeled_train_outputs)
    # softmax = Softmax(10, 200, regularization = 1e-4)
    # softmax.train(softmax_training, \
                  # [], \
                  # 1., \
                  # 1, \
                  # 1, \
                  # LearningMethod('L-BFGS-B', {'max_iter' : 100})
                 # )
    # labeled_test_inputs, labeled_test_outputs = *labeled_test
    # softmax_test_inputs, _, _ = \
        # autoencoder_mnist_network.feedforward(labeled_test_inputs, 1)
    # test_correct = softmax.evaluate(softmax_test_inputs, labeled_test_outputs) * 100.
    # print 'Test set accuracy:', str(test_correct)

if __name__ == '__main__':
    self_taught_learning_mnist()
