import numpy as np
import random 

#-----------------------------------------------------------------------------------------------#
def f(t, X, y, beta, b):
    '''
    *Args: 
        t: evaluation point, shape (n_feature, )
        X: Input data, shape (n_obs, n_feature)
        y: Target label, shape (n_obs, n_feature)
        beta: coefficient vector (n_obs, )
        b:  bias, float 

    Returns:
        f(t): function evaluation, float
    '''
    return b + np.sum(beta * y * (X @ t))
#-----------------------------------------------------------------------------------------------#
def transform_coefficients(X, y, beta, b):
    '''
    Function to transform beta, b into alpha coefficients
    '''
    n_obs = len(y)
    alpha_temp = 0
    for l in range(0,n_obs):
        x_l = X[l].ravel()
        alpha_temp += beta[l]*y[l]*x_l

    alpha = np.zeros(X.shape[1]+1)
    alpha[0] = b
    alpha[1:X.shape[1]+1] = alpha_temp

    return alpha
#-----------------------------------------------------------------------------------------------#

def one_step(X, y, C, beta, b, i, j):
    '''
    OneStep algorithm to update the coefficients beta_i, beta_j and the bias b
    *Args: 
        X: Input data, shape (n_obs, n_features)
        y: Target labels, shape (n_obs,)
        C: Reguarlization Constant, float
        beta: Coefficient vector, shape (n_obs,)
        b: bias, float
        i: Index i, integer
        j: Index i, integer
    Returns: 
        beta: Updated Coefficicent vector, shape (n_obs,)
        b: Updated bias, float
    '''

    # Extract the values of beta for the pair if indices i,j
    beta = beta.copy()
    beta_i = beta[i]
    beta_j = beta[j]
    # Extract the vectors x_i and x_j
    x_i = X[i].ravel()
    x_j = X[j].ravel()
    # Extract the values y_i and y_j
    y_i = y[i]
    y_j = y[j]

    # Error
    error_i = f(x_i, X, y, beta, b) - y_i
    error_j = f(x_j, X, y, beta, b) - y_j
    delta = y_i*(  error_j  - error_i )
    s = y_i*y_j
    chi = np.dot(x_i, x_i) + np.dot(x_j, x_j) - 2*np.dot(x_i, x_j)
    gamma =  s*beta_i + beta_j

    if s==1: 
        L = np.max([0, gamma - C])
        H = np.min([gamma, C])
    else: 
        L = np.max([0, -gamma])
        H = np.min([C,C - gamma])
    
    # Update beta_i
    if chi > 0: 
        beta_i_new = np.min(    [np.max([beta_i + delta/chi, L])   , H])
    elif delta >0: 
        beta_i_new = L
    else: 
        beta_i_new = H

    # Update beta_j with updated beta_i
    beta_j_new = gamma - s*beta_i_new
    
    # Update bias b 
    beta_temp = beta.copy()
    beta_temp[i] = beta_i_new
    beta_temp[j] = beta_j_new

    b_new = b - 0.5 * ((f(x_i, X, y, beta_temp, b) - y_i) + (f(x_j, X, y, beta_temp, b) - y_j))
    b = b_new
    # Insert the new beta values in the beta array 
    beta[i] = beta_i_new
    beta[j] = beta_j_new

    # Return the updated value for beta and b
    return beta, b
#-----------------------------------------------------------------------------------------------#
def SMO(X, y, C, n_iter=10000):

    '''
    Sequential Minimization Algorithm (SMO) for Support Vector Machine
    *Args: 
        X: Input data, shape (n_obs, n_features)
        y: Target labels, shape (n_obs,)
        C: Reguarlization Constant, float
        iter: Number of iterations, integer
    Returns: 
        beta: Coefficicent vector, shape (n_obs,)
        b: bias, float
    '''

    # Set the number of observations
    n_obs = X.shape[0]

    # Initialize the bias and beta
    b = 0
    beta = np.zeros(n_obs)

    n = 0
    while n < n_iter:
        # Draw the indices i,j
        i,j = np.random.choice(n_obs, 2, replace=False)
    
        # Call the one-step update for beta and b
        beta_new, b_new = one_step(X, y, C, beta, b, i, j)
        beta = beta_new
        b = b_new
        n+=1
    
    # Update the bias 
    sv_indices = np.where((beta > 0) & (beta < C))[0]
    if sv_indices.size > 0:
        errors = [f(X[l], X, y, beta, b) - y[l] for l in sv_indices]
        b = b - np.mean(errors)

    return beta, b
#-----------------------------------------------------------------------------------------------#
def predict(X_test, X_train, y_train, beta, b):
    '''
    Function to calculate prediction on a given test set X_test 
    *Args: 
        X_test: Test sample features, (n_test_obs, n_features)
        X_train: Training sample features, shape (n_obs, n_features)
        y_train: Labels for training sample, (n_obs, )
        beta: Coefficient vector (n_obs, )
        b: bias, float
    Returns: 
        predictions_classified: labeled {-1,1} predictions on the test set X_test with decison_boundary = 0
    '''
    n_obs = X_train.shape[0]
    n_obs_test = X_test.shape[0]
    predictions = np.zeros(n_obs_test)

    for i in range(0,n_obs_test):
        x_i = X_test[i].ravel()
        predictions[i] = f(x_i, X_train, y_train, beta, b)

    predictions_classified = np.sign(predictions).astype(int)
    return predictions_classified
#-----------------------------------------------------------------------------------------------#
def confusion_matrix_svm(X_test, y_test, X_train, y_train, beta, b):

    """
    Confusion matrix for classification

    *Args: 
        X: Input data, shape (n_samples, n_features)
        y: Target labels, shape (n_samples,)
        alpha: Coefficient vector (n_features,)
        threshold: Threshold value for decision boundary, float

    Returns: 
        C: Confusion matrix (2,2)
    """

    n_obs_test = X_test.shape[0]
    # Generate the 2x2 confusion matrix
    C = np.zeros((2, 2), dtype=int)

    # Generate the predictions
    predictions = np.zeros(n_obs_test)
    for i in range(0,n_obs_test):
        x_i = X_test[i].ravel()
        predictions[i] = f(x_i, X_train, y_train, beta, b)

    predictions_classified = np.sign(predictions).astype(int)

    for i in range(n_obs_test): 
        if (predictions_classified[i] == -1 and y_test[i]==-1):
            C[0,0] +=1
        if (predictions_classified[i] == -1 and y_test[i]==1):
            C[0,1] +=1
        if (predictions_classified[i] == 1 and y_test[i]==-1):
            C[1,0] +=1
        if (predictions_classified[i] == 1 and y_test[i]==1):
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

def SMO_wKKT(X,y,C,n_iter=10000):

    '''
    Sequential Minimization Algorithm (SMO) for Support Vector Machine
    *Args: 
        X: Input data, shape (n_samples, n_features)
        y: Target labels, shape (n_samples,)
        C: Reguarlization Constant, float
        iter: Number of iterations, integer
    Returns: 
        beta: Coefficicent vector, shape (n_samples,)
        b: bias, float
    '''

    # Set the number of observations
    n_obs = len(y)

    # Initialize the bias and beta
    b = 0
    beta = np.zeros(n_obs)

    # Loop over the number of iterations 
    n = 0
    while n < n_iter:

        # Outer loop iterating over indices
        KKT = np.zeros(n_obs)
        for i in range(0,n_obs):
            x_i = X[i].ravel()
            y_i = y[i]
            # KKT_i condition
            KKT_i = (C-beta[i])*max(0, 1 - y_i*f(x_i, X,y, beta, b) ) + beta[i]*max(0, y_i*f(x_i, X,y, beta, b)-1)
            KKT[i] = KKT_i
            if KKT_i > 0: 

                # Pick the index j:
                margin_indices = list(np.where((C > beta)  & (beta >0))[0])
                if i in margin_indices:
                    margin_indices.remove(i)
                
                if len(margin_indices)>0:
                    j = np.random.choice(margin_indices)
                else:
                    indices = list(range(0, n_obs))
                    indices.remove(i)
                    j = np.random.choice(    indices    )

                # Call the one-step update for beta and b
                beta_new, b_new = one_step(X, y, C, beta, b, i, j)
                beta = beta_new
                b = b_new
        
        if np.sum(KKT)==0:
            break    
        n += 1

    # Update the bias 
    sv_indices = np.where((beta > 0) & (beta < C))[0]
    if sv_indices.size > 0:
        errors = [f(X[l], X, y, beta, b) - y[l] for l in sv_indices]
        b = b - np.mean(errors)

    return beta, b

#------------------------------------------------------------------------------------------'
def f_wkernel(t, K, X, y, beta, b):
    '''
    Function evaluation with Kernel 
    *Args: 
        t: shape (n_feature,)
        K: Kernel function 
        X: Input data, shape (n_obs, n_feature)
        y: Input label, 
        beta: Coefficient vector
        b: bias

    Returns:
        f(t): 
    '''
    n_obs = X.shape[0]
    sum = 0
    for l in range(0,n_obs):
        x_l = X[l].ravel()
        sum = sum + beta[l]*y[l]*K(t, x_l)

    return b + sum   
#------------------------------------------------------------------------------------------'
def one_step_wkernel(K, X, y, C, beta, b, i,j):
    '''
    OneStep algorithm to update the coefficients beta_i, beta_j and the bias b with kernel function
    *Args: 
        K: Kernel function 
        X: Input data, shape (n_samples, n_features)
        y: Target labels, shape (n_samples,)
        C: Reguarlization Constant, float
        beta: Coefficient vector, shape (n_samples,)
        b: bias, float
        i: Index i, integer
        j: Index i, integer
    Returns: 
        beta: Updated Coefficicent vector, shape (n_samples,)
        b: Updated bias, float
    '''

    # Extract the values of beta for the pair if indices i,j
    beta = beta.copy()
    beta_i = beta[i]
    beta_j = beta[j]
    # Extract the vectors x_i and x_j
    x_i = X[i].ravel()
    x_j = X[j].ravel()
    # Extract the values y_i and y_j
    y_i = y[i]
    y_j = y[j]

    # Error
    error_i = f_wkernel(x_i, K, X, y, beta, b) - y_i
    error_j = f_wkernel(x_j, K, X, y, beta, b) - y_j
    delta = y_i*(  error_j  - error_i )
    s = y_i*y_j
    chi = K(x_i, x_i) + K(x_j, x_j) - 2*K(x_i, x_j)
    gamma =  s*beta_i + beta_j

    if s==1: 
        L = np.max([0, gamma - C])
        H = np.min([gamma, C])
    else: 
        L = np.max([0, -gamma])
        H = np.min([C,C - gamma])
    
    if chi > 0: 
        beta_i_new = np.min(    [np.max([beta_i + delta/chi, L])   , H])
    elif delta >0: 
        beta_i_new = L
    else: 
        beta_i_new = H

    beta_j_new = gamma - s*beta_i_new
    
    # Update bias b 
    beta_temp = beta.copy()
    beta_temp[i] = beta_i_new
    beta_temp[j] = beta_j_new

    b_new = b - 0.5 * ((f_wkernel(x_i, K, X, y, beta_temp, b) - y_i) + (f_wkernel(x_j,K, X, y, beta_temp, b) - y_j))
    b = b_new
    # Update the beta values in the array beta 
    beta[i] = beta_i_new
    beta[j] = beta_j_new

    # Return the updated value for beta and b
    return beta, b
#------------------------------------------------------------------------------------------'
def SMO_wkernel(K, X, y, C, n_iter =10000):

    '''
    Sequential Minimization Algorithm (SMO) for Support Vector Machine
    *Args: 
        K: Kernel function 
        X: Input data, shape (n_samples, n_features)
        y: Target labels, shape (n_samples,)
        C: Reguarlization Constant, float
        iter: Number of iterations, integer
    Returns: 
        beta: Coefficicent vector, shape (n_samples,)
        b: bias, float
    '''

    # Set the number of observations
    n_obs = X.shape[0]

    # Initialize the bias and beta
    b = 0
    beta = np.zeros(n_obs)

    n = 0
    while n < n_iter:
        # Draw the indices i,j
        i,j = np.random.choice(n_obs, 2, replace=False)

        # Call the one-step update for beta and b
        beta_new, b_new = one_step_wkernel(K, X, y, C, beta, b, i, j)
        beta = beta_new
        b = b_new
        n+=1

    # Update the bias 
    sv_indices = np.where((beta > 0) & (beta < C))[0]
    if sv_indices.size > 0:
        errors = [f_wkernel(X[l], K, X, y, beta, b) - y[l] for l in sv_indices]
        b = b - np.mean(errors)

    return beta, b