import scipy.io
from learning import LearningMethod
from neural_network import NeuralNetwork, sigmoid, SparsityParams

def linear_decoder_stl():
    identity = lambda x: x
    linear_decoder_network = \
        NeuralNetwork([192, 400, 192], \
                      activation_func=[sigmoid, identity], \
                      regularization=0.003, \
                      sparsity_params=SparsityParams(0.035, 5.) \
                     )
    patches = \
        scipy.io.loadmat('../neural_network_ufldl/linear_decoder_exercise/stlSampledPatches.mat')['patches'].T
    linear_decoder_network.train([patches, patches], \
                                 [], \
                                 1., \
                                 1, \
                                 1, \
                                 learning_method=LearningMethod('L-BFGS-B', {'max_iter' : 1}), \
                                )
   
if __name__ == '__main__':
    linear_decoder_stl()
