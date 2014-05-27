# TODO:
# -in the transform below, is there a material difference between the rotations
#  U * X versus U^T * X if U is orthogonal? I.e., is it possible for the columns
#  of both U and U^T to be eigenvectors for var(X)? seems like no, and there is
#  a material difference - try to understand why. also try to understand how we
#  should interpret the outputs of PCA in different packages - are the columns
#  or the rows the principal components?
# -implement the below using native functions / args of PCA object

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv, sqrtm
from sklearn.decomposition import PCA

def pca_2d():
    raw_data = np.genfromtxt('../neural_network_ufldl/pca_2d/pcaData.txt')
    data = raw_data - np.tile(np.mean(raw_data, 1), (raw_data.shape[1], 1)).T
    
    pca = PCA().fit(data.T)
    # print pca.components_
    # print pca.explained_variance_ratio_

    # raw data + principal components..
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
    covar_rotated = rotated.dot(rotated.T)
    pca_whitened = sqrtm(inv(covar_rotated)).dot(rotated)
    # plt.scatter(pca_whitened[0], pca_whitened[1])
    
    # zca-whitened data..
    zca_whitened = pca.components_.T.dot(pca_whitened)
    plt.scatter(zca_whitened[0], zca_whitened[1])
    
    #plt.axes().set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    pca_2d()
