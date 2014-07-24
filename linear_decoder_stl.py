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

def zca_whiten(patches, regularize=True):
    # first ensure each image is centered around 0
    # for i, patch in enumerate(patches):
        # patches[i] = patch - np.mean(patch)

    # then ensure each pixel is centered around 0
    mean_patch = np.mean(patches, 0)
    patches -= mean_patch

    pca = PCA().fit(patches)
    if not regularize:
        zca_factor = pca.components_.T.dot(PCA(whiten=True).fit(patches).components_)
    else:
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
    return zca_whitened_b, zca_factor, mean_patch

# notes:
# -it's important to do ZCA. without this, you'll still get some edges, but
#  also a lot of features that are all-white / all-black.
# -it's important to regularize the ZCA as well - otherwise, features look like spots
# -curiously, performance was a lot worse for diff hyperparams (using the same
#  ones as for sparse_autoencoder images)
# -for an unsupervised learning exercise like this, it's not clear how to
#  gauge that a run is "good" without visualization. perhaps once you feed
#  its output to a supervised algorithm, you can better evaluate
# -standard non-linear decoder yielded very strange homogeneous features
def linear_decoder_stl():
    patches = \
        scipy.io.loadmat('../neural_network_ufldl/linear_decoder_exercise/stlSampledPatches.mat')['patches'].T
    zca_whitened, zca_factor, _ = zca_whiten(patches)
        
    identity = lambda x: x
    linear_decoder_network = \
        NeuralNetwork([192, 400, 192], \
                      activation_func=[sigmoid, identity], \
                      regularization=0.003, \
                      sparsity_params=SparsityParams(0.035, 5.) \
                     )
    
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

    with open('../neural_network_ufldl/linear_decoder_exercise/linear_decoder_network_backup.pkl', 'wb') as network_file2:
        pickle.dump((linear_decoder_network.biases, linear_decoder_network.weights), \
                     network_file2 \
                   )

    weight = linear_decoder_network.weights[0]
    display_image_grid(weight.dot(zca_factor), 8, 20, rgb=True)
   
if __name__ == '__main__':
    # toy_example()
    linear_decoder_stl()
