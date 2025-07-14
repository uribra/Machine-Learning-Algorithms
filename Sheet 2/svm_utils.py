import numpy as np

def one_step(X, y, beta, b, i, j, C):
    """
    Implementation of Algorithm 2.1 from the document.
    Takes one iterative step of the SMO algorithm for two selected indices i and j.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Training data
    y : array, shape (n_samples,)
        Labels (-1 or 1)
    beta : array, shape (n_samples,)
        Current Lagrange multipliers
    b : float
        Current bias term
    i, j : int
        Indices of the two points to optimize
    C : float
        Regularization parameter
    
    Returns:
    --------
    beta : array, shape (n_samples,)
        Updated Lagrange multipliers
    b : float
        Updated bias term
    """
    n_samples = len(X)

    # function values i and j
    f_x_i = 0
    f_x_j = 0
    for l in range(n_samples):
        f_x_i += beta[l] * y[l] * np.dot(X[l], X[i])
        f_x_j += beta[l] * y[l] * np.dot(X[l], X[j])
    f_x_i += b
    f_x_j += b
    
    delta = y[i] * ((f_x_j - y[j]) - (f_x_i - y[i]))
    
    s = y[i] * y[j]
    
    chi = np.dot(X[i], X[i]) + np.dot(X[j], X[j]) - 2 * np.dot(X[i], X[j])
    
    gamma = s * beta[i] + beta[j]
    
    if s == 1:
        L = max(0, gamma - C)
        H = min(gamma, C)
    else:
        L = max(0, -gamma)
        H = min(C, C - gamma)
    
    # update beta[i]
    if chi > 0:
        beta[i] = min(max(beta[i] + delta/chi, L), H)
    elif delta > 0:
        beta[i] = L
    else:
        beta[i] = H
    
    # update beta[j]
    beta[j] = gamma - s * beta[i]
    
    # update function 
    f_x_i_new = 0
    f_x_j_new = 0
    for l in range(n_samples):
        f_x_i_new += beta[l] * y[l] * np.dot(X[l], X[i])
        f_x_j_new += beta[l] * y[l] * np.dot(X[l], X[j])
    f_x_i_new += b
    f_x_j_new += b

    # update bias term
    b_new = b - 0.5 * (f_x_i_new - y[i] + f_x_j_new - y[j])
    
    return beta, b_new

def smo(X, y, C, n_steps):
    """
    Implementation of the Sequential Minimal Optimization (SMO) algorithm
    to train a Support Vector Machine classifier.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Training data
    y : array, shape (n_samples,)
        Labels (-1 or 1)
    C : float
        Regularization parameter that controls the trade-off between
        maximizing the margin and minimizing the classification error
    n_steps : int
        Number of optimization steps to perform
    
    Returns:
    --------
    beta : array, shape (n_samples,)
        Optimized Lagrange multipliers (dual variables)
    b : float
        Optimized bias term
    """
    n_samples = len(X)
    b = 0
    beta = np.zeros(n_samples)
    
    for step in range(n_steps):
        #why does this change when you draw i and j with Uriels method 
        i, j = np.random.choice(n_samples, 2, replace=False)
        beta, b = one_step(X, y, beta, b, i, j, C)
    
    # calculate mean error for margin support vectors (0 < beta < C)
    margin_sv_indices = []
    for k in range(n_samples):
        if 0 < beta[k] < C:
            margin_sv_indices.append(k)
    
    if margin_sv_indices:
        errors = []
        for k in margin_sv_indices:
            f_x_k = 0
            for l in range(n_samples):
                f_x_k += beta[l] * y[l] * np.dot(X[l], X[k])
            f_x_k += b
            errors.append(f_x_k - y[k])
        
        mean_error = np.mean(errors)
        b = b - mean_error
    
    return beta, b

def classify_svm(X_new, X, y, beta, b):
    """
    Classifies new data points using a trained SVM model.
    
    Parameters:
    -----------
    X_new : array, shape (n_samples, n_features)
        New data points to classify
    X : array, shape (n_train_samples, n_features)
        Training data used to train the SVM
    y : array, shape (n_train_samples,)
        Training labels used to train the SVM
    beta : array, shape (n_train_samples,)
        Optimized Lagrange multipliers from SVM training
    b : float
        Optimized bias term from SVM training
    
    Returns:
    --------
    predicted_labels : array, shape (n_samples,)
        Predicted class labels (-1 or 1)
    """
    predictions = np.zeros(X_new.shape[0])
    
    for i in range(X_new.shape[0]):
        f_x = 0
        for j in range(len(X)):
            if beta[j] > 0:  
                f_x += beta[j] * y[j] * np.dot(X[j], X_new[i])
        f_x += b
        
        predictions[i] = np.sign(f_x)
    
    return predictions

def calculate_kkt(X, y, beta, b, i, C):
    """
    Calculate the KKT condition value for data point i.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Training data
    y : array, shape (n_samples,)
        Labels (-1 or 1)
    beta : array, shape (n_samples,)
        Current Lagrange multipliers
    b : float
        Current bias term
    i : int
        Index of the point to check
    C : float
        Regularization parameter
    
    Returns:
    --------
    kkt_value : float
        The value of the KKT condition for point i
    """
    # Calculate f(x_i)
    f_x_i = 0
    for l in range(len(X)):
        f_x_i += beta[l] * y[l] * np.dot(X[l], X[i])
    f_x_i += b
    
    # Calculate KKT value according to equation (2.3)
    term1 = (C - beta[i]) * max(0, 1 - y[i] * f_x_i)
    term2 = beta[i] * max(0, y[i] * f_x_i - 1)
    kkt_value = term1 + term2
    
    return kkt_value

def smo_with_kkt_heuristic(X, y, C, max_steps):
    """
    Implementation of SMO algorithm with KKT condition heuristic for
    index selection as described in Task 2.5.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Training data
    y : array, shape (n_samples,)
        Labels (-1 or 1)
    C : float
        Regularization parameter
    max_steps : int
        Maximum number of optimization steps to perform
    
    Returns:
    --------
    beta : array, shape (n_samples,)
        Optimized Lagrange multipliers (dual variables)
    b : float
        Optimized bias term
    steps_taken : int
        Number of optimization steps actually performed
    """
    n_samples = len(X)
    b = 0
    beta = np.zeros(n_samples)
    steps_taken = 0
    
    while steps_taken < max_steps:
        kkt_violated = False
        
        for i in range(n_samples):
            # check if KKT condition is violated
            kkt_i = calculate_kkt(X, y, beta, b, i, C)
            
            if kkt_i > 0:
                kkt_violated = True
                
                # find a valid j to pair with i
                valid_js = []
                for j in range(n_samples):
                    if j != i and 0 < beta[j] < C:
                        valid_js.append(j)
                
                if valid_js:
                    j = np.random.choice(valid_js)
                else:
                    remaining_indices = [j for j in range(n_samples) if j != i]
                    j = np.random.choice(remaining_indices)
                
                beta, b = one_step(X, y, beta, b, i, j, C)
                steps_taken += 1
                
                if steps_taken >= max_steps:
                    break
        
        if not kkt_violated:
            break
    
    # calculate mean error for margin support vectors (0 < beta < C)
    margin_sv_indices = []
    for k in range(n_samples):
        if 0 < beta[k] < C:
            margin_sv_indices.append(k)
    
    if margin_sv_indices:
        errors = []
        for k in margin_sv_indices:
            f_x_k = 0
            for l in range(n_samples):
                f_x_k += beta[l] * y[l] * np.dot(X[l], X[k])
            f_x_k += b
            errors.append(f_x_k - y[k])
        
        mean_error = np.mean(errors)
        b = b - mean_error
    
    return beta, b, steps_taken

def one_step_kernel(K, y, beta, b, f_values, i, j, C):
    """
    Takes one step of the SMO algorithm using kernel matrix K.
    
    Parameters:
    -----------
    K : array, shape (n_samples, n_samples)
        Precomputed kernel matrix where K[i, j] = kernel_func(X[i], X[j])
    y : array, shape (n_samples,)
        Labels (-1 or 1)
    beta : array, shape (n_samples,)
        Current Lagrange multipliers
    b : float
        Current bias term
    f_values : array, shape (n_samples,)
        Current function values for all points
    i, j : int
        Indices of the two points to optimize
    C : float
        Regularization parameter
    
    Returns:
    --------
    beta : array, shape (n_samples,)
        Updated Lagrange multipliers
    b : float
        Updated bias term
    f_values : array, shape (n_samples,)
        Updated function values
    """
    # use precomputed function and kernel values
    f_x_i = f_values[i]
    f_x_j = f_values[j]
    
    delta = y[i] * ((f_x_j - y[j]) - (f_x_i - y[i]))
    
    s = y[i] * y[j]
    
    chi = K[i, i] + K[j, j] - 2 * K[i, j]
    
    gamma = s * beta[i] + beta[j]
    
    if s == 1:
        L = max(0, gamma - C)
        H = min(gamma, C)
    else:
        L = max(0, -gamma)
        H = min(C, C - gamma)
    
    old_beta_i = beta[i]
    old_beta_j = beta[j]
    
    if chi > 0:
        beta[i] = min(max(beta[i] + delta/chi, L), H)
    elif delta > 0:
        beta[i] = L
    else:
        beta[i] = H
    
    beta[j] = gamma - s * beta[i]
    
    # the change in beta values
    d_beta_i = beta[i] - old_beta_i
    d_beta_j = beta[j] - old_beta_j
    
    if d_beta_i != 0 or d_beta_j != 0:
        for k in range(len(y)):
            f_values[k] += y[i] * d_beta_i * K[i, k] + y[j] * d_beta_j * K[j, k]
    
    b_old = b
    b = b - 0.5 * (f_values[i] - y[i] + f_values[j] - y[j])
    d_b = b - b_old
    
    if d_b != 0:  
        f_values += d_b
    
    return beta, b, f_values

def smo_kernel(X, y, C, n_steps, kernel_func):
    """
    Sequential Minimal Optimization (SMO) algorithm with kernel support.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Training data
    y : array, shape (n_samples,)
        Labels (-1 or 1)
    C : float
        Regularization parameter
    n_steps : int
        Number of optimization steps
    kernel_func : function
        Kernel function that takes two vectors and returns their kernel value
    
    Returns:
    --------
    beta : array, shape (n_samples,)
        Optimized Lagrange multipliers
    b : float
        Optimized bias term
    """
    n_samples = len(X)
    b = 0
    beta = np.zeros(n_samples)
    
    # precompute kernel values
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])
    
    f_values = np.zeros(n_samples)
    
    for step in range(n_steps):
        i, j = np.random.choice(n_samples, 2, replace=False)
        
        beta, b, f_values = one_step_kernel(K, y, beta, b, f_values, i, j, C)
    
    margin_sv_indices = [k for k in range(n_samples) if 0 < beta[k] < C]
    if margin_sv_indices:
        mean_error = np.mean([f_values[k] - y[k] for k in margin_sv_indices])
        b = b - mean_error
    
    return beta, b