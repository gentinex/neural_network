import cPickle as pickle
import datetime
import numpy as np
import numpy.random as random
import scipy.io
from learning import LearningMethod
from linear_decoder_stl import zca_whiten
from scipy.signal import fftconvolve
from softmax import Softmax

def convert_to_stl_vector(output):
    vector_output = np.zeros(4)
    vector_output[output] = 1.0
    return vector_output

def convolve_and_pool(images, \
                      patches, \
                      patch_bias, \
                      patch_weight, \
                      zca_patch_weight, \
                      zca_mean_patch \
                     ):
    num_images = images.shape[3]
    num_patch_pixels = patches.shape[1] / 3
    patch_dim = int(np.sqrt(num_patch_pixels))
    convolved_dim = num_patch_pixels - patch_dim + 1
    pooling_dim = 19
    pooled_dim = convolved_dim / pooling_dim
    pooled_features = \
        np.zeros((num_images, len(zca_patch_weight), pooled_dim, pooled_dim))
    
    for i in xrange(num_images):
        for j, zca_patch_weight_feature in enumerate(zca_patch_weight):
            convolved_features = np.zeros((3, convolved_dim, convolved_dim))
            bias = patch_bias[j]
            for k in xrange(3):
                image = images[:, :, k, i]
                # convolution:
                # -use calibrated weight as the kernel (for the appropriate channel)
                # -but we need to zca-whiten the patches of the images, so put
                #  this whitening factor into the weight
                weight_raw = \
                    zca_patch_weight_feature[(k * num_patch_pixels):((k + 1) * num_patch_pixels)]
                
                weight = np.fliplr(np.flipud(np.reshape(weight_raw, (patch_dim, patch_dim), 'F')))
                patch_index = patch_dim - 1
                convolved = fftconvolve(weight, image)[patch_index:-patch_index, patch_index:-patch_index]
                # -also, need to zero-center each patch. do this in the
                #  post-convolution stage
                convolved_features[k, :, :] = bias + convolved - zca_mean_patch[j]
            for p in xrange(pooled_dim):
                for q in xrange(pooled_dim):
                    pooled_features[i, j, p, q] = \
                        np.mean(convolved_features[:, \
                                                   (p * pooling_dim):((p + 1) * pooling_dim), \
                                                   (q * pooling_dim):((q + 1) * pooling_dim), \
                                                  ] \
                               )
    return pooled_features
    
# TODO:
# -get it working!
# -better understand pooling
# -visualize train, test images
# -clean up code some more
# -what is the sensitivity to the pooling size we use?
# -what is sensitivity to say max-pooling or other aggregation methods?
# -what would happen if we applied softmax w/o feature learning?
# -or just a straight-up neural network with one hidden layer?
def convolutional_neural_network():
    with open('../neural_network_ufldl/linear_decoder_exercise/linear_decoder_network.pkl', 'rb') as network_file:
        biases, weights = pickle.load(network_file)

    patch_bias = biases[0]
    patch_weight = weights[0]

    patches = \
        scipy.io.loadmat('../neural_network_ufldl/linear_decoder_exercise/stlSampledPatches.mat')['patches'].T
    zca_whitened, zca_factor, mean_patch = zca_whiten(patches)
    zca_patch_weight = patch_weight.dot(zca_factor)
    zca_mean_patch = zca_patch_weight.dot(mean_patch)
    
    train = \
        scipy.io.loadmat('../neural_network_ufldl/cnn_exercise/stlTrainSubset.mat')
    train_images = train['trainImages'] # 64,64,3,2000
    train_labels = train['trainLabels'] # 2000,1
    used_train_labels = \
        np.array([convert_to_stl_vector(output - 1) for output in train_labels])
    print 'pooling training set at', str(datetime.datetime.now())
    pooled_features_train = \
        convolve_and_pool(train_images, \
                          patches, \
                          patch_bias, \
                          patch_weight, \
                          zca_patch_weight, \
                          zca_mean_patch \
                         )
    
    test = \
        scipy.io.loadmat('../neural_network_ufldl/cnn_exercise/stlTestSubset.mat')
    test_images = test['testImages']
    test_labels = test['testLabels']
    used_test_labels = \
        np.array([convert_to_stl_vector(output - 1) for output in test_labels])
    print 'pooling test set at', str(datetime.datetime.now())
    pooled_features_test = \
        convolve_and_pool(test_images, \
                          patches, \
                          patch_bias, \
                          patch_weight, \
                          zca_patch_weight, \
                          zca_mean_patch \
                         )

    random.seed(1)
    pooled_dim = pooled_features_train.shape[-1]
    num_softmax_features = len(zca_patch_weight) * pooled_dim * pooled_dim
    softmax = Softmax(4, num_softmax_features, regularization=1e-4)
    num_train_images = pooled_features_train.shape[0]
    train_inputs = pooled_features_train.reshape(num_train_images, num_softmax_features)
    training = [train_inputs, used_train_labels]
    num_test_images = pooled_features_test.shape[0]
    test_inputs = pooled_features_test.reshape(num_test_images, num_softmax_features)
    test = [test_inputs, used_test_labels]
    softmax.train(training, \
                  test, \
                  1., \
                  1, \
                  1, \
                  LearningMethod('L-BFGS-B', {'max_iter' : 200})
                 )
   
if __name__ == '__main__':
    convolutional_neural_network()
