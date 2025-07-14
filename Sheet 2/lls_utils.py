import numpy as np

def linear_least_squares(X, y):
    """
    Implements linear least squares algorithm.
    X: Input data, shape (n_samples, n_features)
    y: Target labels, shape (n_samples,)
    Returns: coefficient vector alpha
    """
    X_hat = np.column_stack((np.ones(X.shape[0]), X))
    
    XTX = X_hat.T @ X_hat
    XTy = X_hat.T @ y
    alpha = np.linalg.solve(XTX, XTy)
    
    return alpha

def classify_lls(X_new, alpha):
    """
    Classifies new data points using a trained linear least squares model.
    
    Parameters:
    -----------
    X_new : array, shape (n_samples, n_features)
        New data points to classify
    alpha : array, shape (n_features + 1,)
        Coefficients from the linear least squares model
        [alpha_0, alpha_1, alpha_2, ...] where alpha_0 is the bias term
    
    Returns:
    --------
    predicted_labels : array, shape (n_samples,)
        Predicted class labels (-1 or 1)
    """
    # bias term
    X_new_with_bias = np.column_stack((np.ones(X_new.shape[0]), X_new))
    
    predictions = X_new_with_bias @ alpha
    
    predicted_labels = np.sign(predictions)
    
    return predicted_labels