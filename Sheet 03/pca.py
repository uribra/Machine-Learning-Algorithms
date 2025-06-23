import numpy as np

def pca(x_train, x_test, q):
   """
    Compute principal components and the coordinates.
    
    Parameters
    ----------
    
    x_train: (n_train, d) NumPy array
    x_test: (n_test, d) NumPy array
    q: int
       The number of principal components to compute.
       Has to be less than `d`.

    Returns
    -------
    
    Vq: (d, q) NumPy array, orthonormal vectors (column-wise)
    xq_train: (n, q) NumPy array, coordinates for X_train (row-wise) after PCA projection
    xq_test: (n, q) NumPy array, coordinates for X_test (row-wise) after PCA projection 
   """

   # Number of observations in the training data 
   n_train = x_train.shape[0]
   # Dimension of the data set 
   d = x_train.shape[1] 

   # Controll structures for Parameters
   if q >= d:
        raise ValueError('Number of principal components q has to be less than the number of features d')
    
   
   # Calcualte the mean xbar
   xbar = np.mean(x_train, axis = 0)

   # Construct the matrix X with X_i = x_i - xbar
   X_train = x_train - xbar

   # Compute the SVD of X
   _, _, VT = np.linalg.svd(X_train)

   VT = VT[0:q,:]
   Vq = VT.T

   # Compute the new projected coordiates for the data x
   xq_train = np.zeros((n_train,q))
   for i in range(n_train):
      xq_train[i] = np.dot(Vq.T, X_train[i])

   # Compute the projected coordinates for the data x_test from the PCA on the data x
   X_test = x_test - xbar
   # Number of observations for the test data
   n_test = x_test.shape[0]
   xq_test = np.zeros((n_test,q))
   for i in range(n_test):
      xq_test[i] = np.dot(Vq.T, X_test[i]) 

   return Vq, xq_train, xq_test
#----------------------------------------------------------------------------------------------------------#
def percentage_captured_variance(x, q):
    """
    Compute percentage of captured variance 
    
    Parameters
    ----------
    
    x: (n, d) NumPy array
    q: int
       The number of principal components to compute.
       Has to be less than `d`.

    Returns
    -------
    pcv: float
        Captured variance from the first q principal components
   """
    
   # Number of observations in the data 
    n = x.shape[0]
   # Number of features in the data set 
    d = x.shape[1] 
   
   # Controll structures for Parameters
    if q >= d:
        raise ValueError('Number of principal components q has to be less than the number of features d')
    
    # Center the feature data
    xbar = np.mean(x, axis = 0)
    X = x - xbar

    # Compute the singular values of the matrix X
    _, svals,_ = np.linalg.svd(X, full_matrices=False)
    # Compute the eigenvalues of the matrix X*X^T. Note that the eigenvalues of of X*X^T are the squared sigular values of X
    eigvals = svals**2

   # Sum the eigenvalues up to the q-th element
    sum = 0
    for i in range(q):
        sum += eigvals[i]

    pcv = sum / np.sum(eigvals)
    return pcv
#----------------------------------------------------------------------------------------------------------#