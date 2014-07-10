import cPickle as pickle
import numpy as np
import scipy.io
from images import display_image_grid
from learning import LearningMethod
from neural_network import NeuralNetwork, sigmoid, SparsityParams
from sklearn.decomposition import PCA

def toy_example():
    import numpy.random as random
    data = random.randn(10, 5)
    identity = lambda x: x
    linear_decoder_network = \
        NeuralNetwork([5, 8, 5], \
                      activation_func=[sigmoid, identity], \
                      regularization=0.003, \
                      sparsity_params=SparsityParams(0.035, 5.) \
                     )
    print linear_decoder_network.cost_deriv(data, data)
    print linear_decoder_network.backpropagate(data, data)

def zca_whiten(patches):
    # first ensure each image is centered around 0
    # for i, patch in enumerate(patches):
        # patches[i] = patch - np.mean(patch)

    # then ensure each pixel is centered around 0
    patches -= np.mean(patches, 0)

    pca = PCA().fit(patches)
    # pca_whitened = PCA(whiten=True).fit(patches).components_.dot(patches.T)
    #zca_whitened = pca.components_.T.dot(pca_whitened).T
    # return zca_whitened
    
    rotated = pca.components_.dot(patches.T)
    covar_rotated = np.cov(rotated)
    whitening_factor = np.zeros(covar_rotated.shape)
    EPSILON = 0.1
    for (i, j), _ in np.ndenumerate(covar_rotated):
        if i == j:
            whitening_factor[i, j] = 1. / np.sqrt(covar_rotated[i, j] + EPSILON)
    pca_whitened_b = whitening_factor.dot(rotated)

    zca_factor = pca.components_.T.dot(whitening_factor).dot(pca.components_)
    zca_whitened_b = zca_factor.dot(patches.T).T
    #zca_whitened_b = pca.components_.T.dot(pca_whitened_b).T
    return zca_whitened_b, zca_factor

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
    zca_whitened, zca_factor = zca_whiten(patches)
    linear_decoder_network.train([zca_whitened, zca_whitened], \
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

    display_image_grid(weight.dot(zca_factor), 8, 20, rgb=True)
    weight = linear_decoder_network.weights[0]
   
if __name__ == '__main__':
    toy_example()
    # linear_decoder_stl()
