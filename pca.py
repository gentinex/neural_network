import copy
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import scipy.io
from images import display_image, display_image_grid, generate_random_image_slice
from scipy.linalg import inv, sqrtm
from sklearn.decomposition import PCA

def pca_2d():
    raw_data = np.genfromtxt('../neural_network_ufldl/pca_2d/pcaData.txt')
    data = raw_data - np.tile(np.mean(raw_data, 1), (raw_data.shape[1], 1)).T
    
    pca = PCA().fit(data.T)
    # print pca.components_
    # print pca.explained_variance_ratio_

    # raw data + prin3cipal components..
    # plt.scatter(data[0], data[1])
    # plt.plot([0, pca.components_[0, 0]], [0, pca.components_[0, 1]])
    # plt.plot([0, pca.components_[1, 0]], [0, pca.components_[1, 1]])

    # rotated data..
    rotated = pca.components_.dot(data)
    # plt.scatter(rotated[0], rotated[1])

    # dimension-reduced data..
    reduced = pca.components_.T[:, 0:1].dot(pca.components_[0:1, :].dot(data))
    # plt.scatter(reduced[0], reduced[1])
    
    # pca-whitened data..
    covar_rotated = np.cov(rotated)
    whitening_factor = np.zeros(covar_rotated.shape)
    EPSILON = 1e-5
    for (i, j), _ in np.ndenumerate(covar_rotated):
        if i == j:
            whitening_factor[i, j] = 1. / np.sqrt(covar_rotated[i, j] + EPSILON)
    pca_whitened = whitening_factor.dot(rotated)
    # plt.scatter(pca_whitened[0], pca_whitened[1])
    
    # zca-whitened data..
    zca_whitened = pca.components_.T.dot(pca_whitened)
    plt.scatter(zca_whitened[0], zca_whitened[1])
    
    #plt.axes().set_aspect('equal')
    plt.show()
    
def pca_exercise():
    images = scipy.io.loadmat('../neural_network_ufldl/pca_exercise/IMAGES_RAW.mat')['IMAGESr']
    random.seed(100)
    # for i in xrange(images.shape[2]):
        # display_image(images[:, :, i])
        
    # first ensure each image is centered around 0
    image_slices = np.array([generate_random_image_slice(images, 12, 12) for i in xrange(10000)])
    for i, image_slice in enumerate(image_slices):
        image_slices[i] = image_slice - np.mean(image_slice)

    # then ensure each pixel is centered around 0
    image_slices = image_slices - np.tile(np.mean(image_slices, 0), (image_slices.shape[0], 1))
        
    pca = PCA().fit(image_slices)

    # rotate data to orthogonal basis
    rotated = pca.components_.dot(image_slices.T)
    rotated_b = PCA().fit_transform(image_slices).T
    rotated_c = pca.transform(image_slices).T
    #print np.cov(rotated) # in the rotated basis, covar matrix is diagonal
    var_sum = np.cumsum(pca.explained_variance_ratio_)
    k_99 = next((i for i, v in enumerate(var_sum) if v >= 0.99), -1)
    k_90 = next((i for i, v in enumerate(var_sum) if v >= 0.90), -1)

    # produce dimensionally reduced data (note that if the data isn't zero-centered,
    # the transform and inverse_transform functions automatically take care of
    # adjusting back to the feature means)
    reduced_99 = pca.components_.T[:, 0:k_99].dot(pca.components_[0:k_99, :].dot(image_slices.T)).T
    pca_99 = PCA(k_99).fit(image_slices)
    reduced_99_b = pca_99.components_.T.dot(pca_99.components_.dot(image_slices.T)).T
    reduced_99_c = pca_99.inverse_transform(pca_99.transform(image_slices))

    reduced_90 = pca.components_.T[:, 0:k_90].dot(pca.components_[0:k_90, :].dot(image_slices.T)).T
    pca_90 = PCA(k_90).fit(image_slices)

    # display_image_grid(image_slices[0:36], 12, 6)
    # display_image_grid(reduced_99[0:36], 12, 6)
    # display_image_grid(reduced_90[0:36], 12, 6)

    # whiten data, so that covar matrix becomes the identity.
    # note that when regularization is on (i.e., EPSILON is nonzero), one doesn't
    # expect the native whitening to be the same as the manually constructed one
    covar_rotated = np.cov(rotated)
    whitening_factor = np.zeros(covar_rotated.shape)
    EPSILON = 0.1
    for (i, j), _ in np.ndenumerate(covar_rotated):
        if i == j:
            whitening_factor[i, j] = 1. / np.sqrt(covar_rotated[i, j] + EPSILON)
    pca_whitened = whitening_factor.dot(rotated)
    pca_whitened_b = PCA(whiten=True).fit(image_slices).components_.dot(image_slices.T)

    # zca whiten: transform pca whitened data back to original data space
    zca_whitened = pca.components_.T.dot(pca_whitened).T
    display_image_grid(image_slices[0:36], 12, 6)
    display_image_grid(zca_whitened[0:36], 12, 6)

if __name__ == '__main__':
    #pca_2d()
    pca_exercise()
