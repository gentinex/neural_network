import numpy as np
import numpy.random as random
import scipy.io
from images import display_image_grid, generate_random_image_slice, normalize_image_slices
from neural_network import LearningMethod, NeuralNetwork, SparsityParams

''' exercise from UFLDL tutorial: use a sparse autoencoder to come up with
    a simplified representation of input images. a good result consists of
    weights that represent boundaries in the images. note that for this exercise,
    the original MATLAB implementation used L-BFGS-B as the optimization
    algorithm, and that gradient descent seems to work terribly - not clear why.
    there is a doc from andrew ng and co. that says gradient descent performs
    worst on sparse autoencoder problems, but it generally seems to say that it
    takes a lot longer to converge, not that it will converge to a bad value.
    also note that you get bad results if you don't regularize, or set zero
    weight to sparsity. however, it does seem less sensitive to the weights on
    the sparsity cost, or the magnitude of the sparsity param itself (except if
    the sparsity param is too small). '''
def sparse_autoencoder():
    images = scipy.io.loadmat('../neural_network_ufldl/sparseae_exercise/IMAGES.mat')['IMAGES']
    random.seed(100)
    image_slices = np.array([generate_random_image_slice(images, 8, 8) for i in xrange(10000)])
    normalized_image_slices = normalize_image_slices(image_slices)
    autoencoder_network = \
        NeuralNetwork([64, 25, 64], \
                      regularization=0.0001, \
                      sparsity_params=SparsityParams(0.01, 3.) \
                     )
    autoencoder_network.train([normalized_image_slices, normalized_image_slices], \
                            [], \
                            [], \
                            1., \
                            1, \
                            1, \
                            learning_method=LearningMethod('L-BFGS-B', {'max_iter' : 400}), \
                           )
    weight = autoencoder_network.weights[0]
    display_image_grid(weight, 8, 5)

if __name__ == '__main__':
    sparse_autoencoder()
