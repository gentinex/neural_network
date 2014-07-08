import cPickle as pickle
import numpy as np
import scipy.io
from images import display_image_grid
from learning import LearningMethod
from neural_network import NeuralNetwork, sigmoid, SparsityParams

def linear_decoder_stl():
    patches = \
        scipy.io.loadmat('../neural_network_ufldl/linear_decoder_exercise/stlSampledPatches.mat')['patches'].T
      
    identity = lambda x: x
    linear_decoder_network = \
        NeuralNetwork([192, 400, 192], \
                      activation_func=[sigmoid, identity], \
                      regularization=0.003, \
                      sparsity_params=SparsityParams(0.035, 5.) \
                     )
    linear_decoder_network.train([patches, patches], \
                                 [], \
                                 1., \
                                 1, \
                                 1, \
                                 learning_method=LearningMethod('L-BFGS-B', {'max_iter' : 400}), \
                                )

    with open('../neural_network_ufldl/linear_decoder_exercise/linear_decoder_network.pkl', 'wb') as network_file:
        pickle.dump((linear_decoder_network.biases, linear_decoder_network.weights), \
                     network_file \
                   )

    weight = linear_decoder_network.weights[0]
    display_image_grid(weight, 8, 20, rgb=True)
   
if __name__ == '__main__':
    linear_decoder_stl()
