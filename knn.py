import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------#
def kNN_classifier(X_test, X_train, y_train, k):
    '''
    k-nearest neigbour alorithm
    *Args:
        X_test: Input data for test data set, shape (n_obs_test, n_features)
        y_test: Input Labels for test data set, shape (n_obs_train, )
        X_train: Input data for training data set, shape (n_obs_train, n_features)
        y_train: Input Labels, shape (n_obs_train, )
        k: Number of nearest neigbours, integer
    Returns: 
        y_predicted: Predicted labels for the test data set X_test, shape (n_obs_test, )
        near_neigh: nearNeigh_k() value for the test data set X_test, (n_obs_test, )

    Comment: if X_test = X_train then choose k+1 for the k nearest neighbour
    '''

    # Control: if the test data is the training data use k+1
    if np.array_equal(X_test, X_train):
        k_use = k + 1
    else:
        k_use = k

    # Set dimensions of the test and training set 
    n_obs_test = X_test.shape[0]
    n_obs_train = X_train.shape[0]

    # Compute the distances between all the points from the test X_test and the training set X_train
    distances = scipy.spatial.distance.cdist(X_test, X_train, metric = 'euclidean')
    # Extract the indices for the k nearest points for each point in the test data set X_test
    # Note: if X_test = X_train then we each point is counted as a neignbour of itself
    indices_neighbours = np.argpartition(distances, kth = k_use, axis = 1)[:,:k_use]

    # Extract the labels for the neighbours for each point
    y_train_neigh = y_train[indices_neighbours]
    # Calculate the nearNeigh_k(x) value
    near_neigh = np.mean(y_train_neigh, axis = 1)
    # Classify x
    y_predicted_label = np.zeros(n_obs_test).astype(int)
    for i in range(n_obs_test):
        if near_neigh[i]>=0.5:
            y_predicted_label[i] = 1
        else: 
            y_predicted_label[i] = 0

    return y_predicted_label, near_neigh
#-----------------------------------------------------------------------------------------------#
def confusion_matrix_knn(y, y_predicted):

    """
    Confusion matrix for kNN Classification 

    *Args: 
        y: Target labels, shape (n_obs,)
        y_predicted: Predicted target labels, shape(n_obs, )

    Returns: 
        C: Confusion matrix (n_features,n_features)
    """

    # number of observations
    n = len(y)
    # Generate the 2x2 confusion matrix
    C = np.zeros((2, 2), dtype=int)

    for i in range(n): 
        if (y_predicted[i] == 0 and y[i]==0):
            C[0,0] +=1
        if (y_predicted[i] == 0 and y[i]==1):
            C[0,1] +=1
        if (y_predicted[i] == 1 and y[i]==0):
            C[1,0] +=1
        if (y_predicted[i] == 1 and y[i]==1):
            C[1,1] +=1

    return C
#-----------------------------------------------------------------------------------------------#
def accuracy(C):
    """
    Accuracy for from confusion matrix for LLS classification

    *Args: 
        C: Confusion matrix, shape (n_features, n_features)
    Returns: 
        accuracy, float
    """

    acc = np.trace(C)/np.sum(C)
    return acc

#-----------------------------------------------------------------------------------------------#
def kNN_decision_plot(X_test, y_test, X_train, y_train, k):
    '''
    Plots the desicion boundary for the kNN-Algorithm
    *Args:
        X_test: Input data for test data set, shape (n_obs_test, n_features)
        y_test: Input Labels for test data set, shape (n_obs_train, )
        X_train: Input data for training data set, shape (n_obs_train, n_features)
        y_train: Input Labels, shape (n_obs_train, )
        k: Number of nearest neigbours, integer
    Returns: 

    '''

    # Obtain the indices for the labels y=0 and y=1
    indices_0_test = np.where(y_test==0)[0]
    indices_1_test = np.where(y_test==1)[0]

    # Set parameters for scatter and contour plots
    x1_min, x1_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    x2_min, x2_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), 
                         np.linspace(x2_min, x2_max, 100))
    # Concatenate along vertical axis
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]

    _, near_neigh_new = kNN_classifier(grid_points, X_train, y_train, k =k)
    Z = near_neigh_new.reshape(xx1.shape)

    # Scatter plot of the test data
    plt.figure()
    plt.scatter(X_test[indices_0_test,0], X_test[indices_0_test,1], color = "blue", label = "Class 0")
    plt.scatter(X_test[indices_1_test,0], X_test[indices_1_test,1], color = "red", label = "Class 1")
    plt.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=2)
    plt.contourf(xx1, xx2, Z, levels=[0, 0.5, 1], colors=['skyblue', 'salmon'], alpha=0.3)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'k-NN Classification for k = {k}')
    plt.legend()
    plt.show()
    #-----------------------------------------------------------------------------------------------#