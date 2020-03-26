import scipy
import numpy as np
import itertools
import matplotlib.pyplot as plt

import scipy.io as sio
data = sio.loadmat('clusterdata.mat')
X = np.vstack([data['X1'], data['X2'], data['X3']]);
print(X.shape)
# Subtract the mean from each dimension (centering)
m = np.mean(X , axis=0)
X_centered = X - np.tile(m, (X.shape[0], 1))

# Calculate the covariance matrix of the data;
C = np.cov(X_centered.T)

# PCA (or equivalently SVD or EVD) SVD and EVD are equaivalent since C is symmetric PSD
U,S,V = np.linalg.svd(C)

# Project the data onto the first principal component, then back into 2D space
X_recon = np.outer(X_centered.dot(U[:,0]), V[:, 0].T)


#plot centered data and its reconstruction
plt.plot(X_centered[:, 0], X_centered[:, 1], 'o')
plt.plot(X_recon[:, 0], X_recon[:, 1], 'x')

#plot original data and its reconstruction
plt.plot(X[:, 0], X[:, 1], '.')
plt.plot(X_recon[:, 0] + m[0], X_recon[:, 1] + m[1], '.')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.show()
