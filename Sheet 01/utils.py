import numpy as np
import random 

#-----------------------------------------------------------------------------------------------#
def gradient_descent_lls(X, y, nu, eps =  1e-6, maxiter = 1000):

    '''
    Gradient descent algorithm

    *Args: 
        X: Input data, shape (n_samples, n_features)
        y: Target labels, shape (n_samples,)
        nu: 
        eps:
        maxiter:

    Returns: 
        alpha: Coefficient vector alpha shape (n_features + 1,)
        J_values:
        k:
        conv_indi:

    '''

    n_obs = len(y)
    n_features = X.shape[1]
    
    # Define the objective function J
    def J(X, y, alpha):

        n_obs = len(y)
        n_features = X.shape[1]

        X_hat = np.zeros((n_obs, 1 + n_features))
        X_hat[:,0] = 1
        X_hat[:,1:n_features+1] = X

        y_predicted = np.dot(X_hat, alpha)
        J = (1/n_obs)*np.sum((y_predicted - y)**2)
        return J
    
    # Define the gradient of the objective function J
    def gradJ(X, y, alpha):
        n_obs = len(y)
        n_features = X.shape[1]

        X_hat = np.zeros((n_obs, 1 + n_features))
        X_hat[:,0] = 1
        X_hat[:,1:n_features+1] = X
        y_predicted = np.dot(X_hat, alpha)

        gradJ = (2/n_obs)*np.dot(X_hat.T, y_predicted - y)
        return gradJ
    
    # Generate the initial value
    np.random.seed(42)
    alpha_0 = np.random.randn(n_features + 1)
    # Store J values
    J_values = []
    J_values.append(J(X, y, alpha_0))
    # Convergence inidicator
    conv_indi = False

    k = 0
    alpha = alpha_0

    while np.linalg.norm( gradJ(X, y, alpha)) > eps and k < maxiter:
        grad_J = gradJ(X, y, alpha)
        # Updating step
        alpha_new = alpha - nu*grad_J
        k +=1
        alpha = alpha_new
        J_values.append(J(X, y, alpha))
        if np.linalg.norm( gradJ(X, y, alpha)) <= eps:
            conv_indi = True

    return alpha, J_values, conv_indi, k
#-----------------------------------------------------------------------------------------------#
