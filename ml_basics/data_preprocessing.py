# Using simple code to show normalization and PCA


import numpy as np
#given m samples of n dimensional data
n=500
m=1000
X=np.random.randn(m,n)
#normalize within multiple samples of each feature, not across multiple features
X-=np.mean(X, axis=0)

#PCA
cov= np.dot(X.T, X)/(X.shape[0]-1)
U,S,V=np.linalg.svd(cov)
k=100
#X_reduced is a m*k matrix
X_reduced=np.dot(X,U[:,0:k])

print ('ok')