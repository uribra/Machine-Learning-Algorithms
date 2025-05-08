import numpy as np
import random 
import matplotlib.pyplot as plt

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
def confusion_matrix(X, y, alpha, threshold):

    """
    Confusion matrix for LLS classification

    *Args: 
        X: Input data, shape (n_samples, n_features)
        y: Target labels, shape (n_samples,)
        alpha: Coefficient vector (n_features,)
        threshold: Threshold value for decision boundary, float

    Returns: 
        C: Confusion matrix (n_features,n_features)
    """

    # number of observations
    n = len(y)
    # Generate the 2x2 confusion matrix
    C = np.zeros((2, 2), dtype=int)

    # Construct the modified input data 
    X_hat = np.zeros((X.shape[0], 1 + X.shape[1]))
    X_hat[:,0] = 1
    X_hat[:, 1:X.shape[1] + 1] = X

    # Generate the predicted values 
    y_predicted = np.dot(X_hat, alpha)
    y_prediced_binary = y_predicted.copy()

    # Classify 
    y_prediced_binary[y_predicted >= threshold]  = 1
    y_prediced_binary[y_predicted < threshold]  = 0
    
    # Change the data type to integer
    y_prediced_binary = y_prediced_binary.astype(int)

    for i in range(n): 
        if (y_prediced_binary[i] == 0 and y[i]==0):
            C[0,0] +=1
        if (y_prediced_binary[i] == 0 and y[i]==1):
            C[0,1] +=1
        if (y_prediced_binary[i] == 1 and y[i]==0):
            C[1,0] +=1
        if (y_prediced_binary[i] == 1 and y[i]==1):
            C[1,1] +=1

    return C
#-----------------------------------------------------------------------------------------------#
def accuracy(C):
    """
    Accuracy for from confusion matrix for LLS classification

    *Args: 
        C: Confusion matrix, shape (n_features, n_features)
        n: number of observations
    Returns: 
        accuracy, float
    """

    acc = np.trace(C)/np.sum(C)
    return acc
#-----------------------------------------------------------------------------------------------#
def lls_decision_plot(X_test, y_test, alpha, threshold):
    '''
    Plots the desicion boundary for the linear least squares algorithm
    *Args:
        X_test: Input data for test data set, shape (n_obs_test, n_features)
        y_test: 
    Returns: 
        Scatter plot with separting hyperplane
    '''

    # Obtain the indices for the labels y=0 and y=1
    indices_0_test = np.where(y_test==0)[0]
    indices_1_test = np.where(y_test==1)[0]
    x1_min = np.min(X_test[:,0])
    x1_max = np.max(X_test[:,0])
    x2_min = np.min(X_test[:,1])
    x2_max = np.max(X_test[:,1])

    # Generate the meshgrid for hyperplane 
    x1_vals, x2_vals = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                       np.linspace(x2_min, x2_max, 300))

    Z = alpha[0] + alpha[1] * x1_vals + alpha[2] * x2_vals

    # Scatter plot with separating hyperplane
    plt.figure()
    plt.scatter(X_test[indices_0_test,0], X_test[indices_0_test,1], color = "blue", label = "Class 0")
    plt.scatter(X_test[indices_1_test,0], X_test[indices_1_test,1], color = "red", label = "Class 1")
    plt.contour(x1_vals, x2_vals, Z, levels = [threshold], colors='k', linestyles='-', linewidths=2)
    plt.contourf(x1_vals, x2_vals, Z, levels=[-np.inf, threshold, np.inf], colors=['lightblue', 'lightcoral'], alpha=0.5)
    plt.annotate(
        'H',                # Text label
        xy=(-1.25, -2),     # Point to annotate
        xytext=(-1.25,-2),  # Location of text
        #arrowprops=dict(facecolor='black', arrowstyle='->'),  # Arrow style
        fontsize=12
    )
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Hyperplane H: $\\alpha_0 + \\alpha_1 \\cdot x_1 + \\alpha_2 \\cdot x_2 = threshold$')
    plt.legend()
    plt.show()