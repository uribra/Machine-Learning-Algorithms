import numpy as np

def compute_binary_confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix for binary classification with labels -1 and 1.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (-1 or 1)
    y_pred : array-like
        Predicted labels (-1 or 1)
    
    Returns:
    --------
    confusion_matrix : array, shape (2, 2)
        Confusion matrix where:
        [0,0] = True Negatives (predicted -1, actual -1)
        [0,1] = False Positives (predicted 1, actual -1)
        [1,0] = False Negatives (predicted -1, actual 1)
        [1,1] = True Positives (predicted 1, actual 1)
    accuracy : float
        Trace / Sum
    """
    confusion_matrix = np.zeros((2, 2), dtype=int)
    
    for i in range(len(y_true)):
        if y_pred[i] == -1 and y_true[i] == -1:
            confusion_matrix[0, 0] += 1
        elif y_pred[i] == 1 and y_true[i] == -1:
            confusion_matrix[0, 1] += 1 
        elif y_pred[i] == -1 and y_true[i] == 1:
            confusion_matrix[1, 0] += 1
        elif y_pred[i] == 1 and y_true[i] == 1:
            confusion_matrix[1, 1] += 1 
    
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    return confusion_matrix, accuracy

def phi_transform(X):
    """Transform 2D data with the feature map phi(t) = [t₁, t₂, t₁² + t₂²]"""
    n_samples = X.shape[0]
    X_transformed = np.zeros((n_samples, 3))
    X_transformed[:, 0] = X[:, 0]
    X_transformed[:, 1] = X[:, 1]
    X_transformed[:, 2] = X[:, 0]**2 + X[:, 1]**2
    return X_transformed