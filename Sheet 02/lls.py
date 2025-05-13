
import numpy as np
import random 

#-----------------------------------------------------------------------------------------------#
def linear_least_squares(X ,y):
    """
    Linear least squares (LLS) algorithm

    *Args: 
        X: Input data, shape (n_obs, n_features)
        y: Target labels, shape (n_obs,)
    Returns: 
        alpha: Coefficient vector alpha, shape (n_features + 1,)
    """

    # Construct the modified input data (including the constant) 
    X_hat = np.zeros((X.shape[0], 1 + X.shape[1]))
    X_hat[:,0] = 1
    X_hat[:,1:X.shape[1]+1] = X

    # Calculate X^T*X
    XTX = np.dot(X_hat.T, X_hat)
    # Calculate X^T*y
    XTy = np.dot(X_hat.T,y)
    # Solve the linear system (X^T*X)*alpha = X^T*y
    alpha = np.linalg.solve(XTX,XTy)
    return alpha
#-----------------------------------------------------------------------------------------------#
def confusion_matrix_lls(X, y, alpha, threshold):

    """
    Confusion matrix for classification

    *Args: 
        X: Input data, shape (n_samples, n_features)
        y: Target labels, shape (n_samples,)
        alpha: Coefficient vector (n_features,)
        threshold: Threshold value for decision boundary, float

    Returns: 
        C: Confusion matrix (n_features,n_features)
    """

    # number of observations
    n_obs = X.shape[0]
    # Number of features
    n_features = X.shape[1]
    
    # Generate the 2x2 confusion matrix
    C = np.zeros((2, 2), dtype=int)

    # Construct the modified input data 
    X_hat = np.zeros((n_obs, 1 + n_features))
    X_hat[:,0] = 1
    X_hat[:, 1:n_features + 1] = X

    # Generate the predicted values 
    y_predicted = np.dot(X_hat, alpha)
    y_prediced_classified = y_predicted.copy()

    # Classify 
    y_prediced_classified[y_predicted >= threshold]  = 1
    y_prediced_classified[y_predicted < threshold]  = -1
    
    # Change the data type to integer
    y_prediced_classified = y_prediced_classified.astype(int)

    for i in range(n_obs): 
        if (y_prediced_classified[i] == -1 and y[i]==-1):
            C[0,0] +=1
        if (y_prediced_classified[i] == -1 and y[i]==1):
            C[0,1] +=1
        if (y_prediced_classified[i] == 1 and y[i]==-1):
            C[1,0] +=1
        if (y_prediced_classified[i] == 1 and y[i]==1):
            C[1,1] +=1

    return C
#-----------------------------------------------------------------------------------------------#
def accuracy(C):

    """
    Accuracy for from confusion matrix for classification

    *Args: 
        C: Confusion matrix, shape (n_features, n_features)
    Returns: 
        accuracy, float
    """

    acc = np.trace(C)/np.sum(C)
    return acc
#-----------------------------------------------------------------------------------------------#