import numpy as np
import matplotlib.pyplot as plt
from data_analysis_utils import phi_transform

def plot_svm_results_2(X, y, beta, b, C, title='SVM ClassifierPP', ax=None, figsize=(7, 5), zoom=False, print_stats=True):
    """
    Plots the results of SVM classification.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Input data
    y : array, shape (n_samples,)
        Target labels (-1 or 1)
    beta : array, shape (n_samples,)
        Optimized Lagrange multipliers from SVM training
    b : float
        Optimized bias term from SVM training
    C : float
        Regularization parameter for SVM
    title : str, optional
        Title for the plot
    ax : matplotlib.axes.Axes, optional
        If provided, plot on this axis instead of creating a new figure
    figsize : tuple, optional
        Figure size (width, height) in inches, only used if ax is None
    zoom : bool, optional
        Whether to zoom in on a specific region (0-2.5 for t_1, 0-3 for t_2)
    print_stats : bool, optional
        Whether to print support vector statistics
    """
    # xreate figure if axis is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_plot = True
    else:
        show_plot = False
    
    if zoom:
        x_min, x_max = 0, 2.5
        y_min, y_max = 0, 3
    else:
        margin = 1.5
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    
    h = 0.02  
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.zeros(mesh_points.shape[0])

    # calculate decision for mesh 
    for i in range(len(X)):
        if beta[i] > 0:  
            Z += beta[i] * y[i] * np.dot(X[i], mesh_points.T)
    Z += b
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=[-float('inf'), 0, float('inf')], 
                colors=['skyblue', 'salmon'], alpha=0.4)
    
    cs = ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    ax.clabel(cs, inline=1, fontsize=10, fmt='f = 0')
    
    # find support vectors and margin vectors
    support_idx = beta > 0
    margin_idx = (beta > 0) & (beta < C)
    regular_neg_idx = (y == -1) & ~support_idx
    regular_pos_idx = (y == 1) & ~support_idx
    
    # plot non-support vectors
    ax.scatter(X[regular_neg_idx, 0], X[regular_neg_idx, 1], c='blue', marker='o', 
              label='Class -1', s=20, alpha=0.7)
    ax.scatter(X[regular_pos_idx, 0], X[regular_pos_idx, 1], c='red', marker='o', 
              label='Class 1', s=20, alpha=0.7)
    
    # plot support vectors as diamonds
    sv_neg_idx = support_idx & (y == -1) & ~margin_idx
    sv_pos_idx = support_idx & (y == 1) & ~margin_idx
    ax.scatter(X[sv_neg_idx, 0], X[sv_neg_idx, 1], c='blue', marker='D', 
              s=30, label='Support Vector', alpha=0.7)
    ax.scatter(X[sv_pos_idx, 0], X[sv_pos_idx, 1], c='red', marker='D', 
              s=30, alpha=0.7)
    
    # plot margin vectors as stars
    mv_neg_idx = margin_idx & (y == -1)
    mv_pos_idx = margin_idx & (y == 1)
    ax.scatter(X[mv_neg_idx, 0], X[mv_neg_idx, 1], c='blue', marker='*', 
              s=40, label='Margin Vectors', alpha=0.7)
    ax.scatter(X[mv_pos_idx, 0], X[mv_pos_idx, 1], c='red', marker='*', 
              s=40, alpha=0.7)
    
    zoom_text = " (Zoomed)" if zoom else ""
    ax.set_title(f'{title}{zoom_text}')
    ax.set_xlabel('$t_1$')
    ax.set_ylabel('$t_2$')
    ax.legend(loc='upper right', fontsize='small')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    
    if show_plot:
        plt.tight_layout()
        plt.show()
    
    if print_stats and not zoom:
        print(f"Number of support vectors for C={C}: {np.sum(support_idx)}")
        print(f"Number of margin defining vectors for C={C}: {np.sum(margin_idx)}")
    
    return ax

def plot_gaussian_kernel_svm(X, y, beta, b, C, sigma=1.0, title='SVM with Gaussian Kernel', ax=None, figsize=(7, 5), print_stats=True):
    """
    Plots the results of SVM classification with a Gaussian kernel.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Original input data
    y : array, shape (n_samples,)
        Target labels (-1 or 1)
    beta : array, shape (n_samples,)
        Optimized Lagrange multipliers from SVM training
    b : float
        Optimized bias term from SVM training
    C : float
        Regularization parameter for SVM
    sigma : float, optional
        Width parameter for the Gaussian kernel
    title : str, optional
        Title for the plot
    ax : matplotlib.axes.Axes, optional
        If provided, plot on this axis instead of creating a new figure
    figsize : tuple, optional
        Figure size (width, height) in inches, only used if ax is None
    print_stats : bool, optional
        Whether to print support vector statistics
    """
    
    # new figure if axis is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_plot = True
    else:
        show_plot = False
    
    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    
    h = 0.02 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = np.zeros(mesh_points.shape[0])
    
    sv_indices = np.where(beta > 0)[0]
    
    for i in sv_indices:
        # Calculate kernel between X[i] and all mesh points at once
        kernel_values = np.exp(-np.sum((mesh_points - X[i])**2, axis=1) / (2 * sigma**2))
        Z += beta[i] * y[i] * kernel_values
    
    Z += b
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=[-float('inf'), 0, float('inf')], 
                colors=['skyblue', 'salmon'], alpha=0.4)
    
    cs = ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    ax.clabel(cs, inline=1, fontsize=10, fmt='f = 0')
    
    support_idx = beta > 0
    
    regular_neg_idx = (y == -1) & ~support_idx
    regular_pos_idx = (y == 1) & ~support_idx
    
    boundary_sv_idx = (beta > 0) & (np.isclose(beta, C) | (beta >= C))
    
    margin_sv_idx = (beta > 0) & (beta < C) & ~np.isclose(beta, C)
    
    ax.scatter(X[regular_neg_idx, 0], X[regular_neg_idx, 1], c='blue', marker='o', 
              label='Class -1', s=20, alpha=0.7)
    ax.scatter(X[regular_pos_idx, 0], X[regular_pos_idx, 1], c='red', marker='o', 
              label='Class 1', s=20, alpha=0.7)
    
    boundary_sv_neg_idx = boundary_sv_idx & (y == -1)
    boundary_sv_pos_idx = boundary_sv_idx & (y == 1)
    if np.any(boundary_sv_neg_idx):
        ax.scatter(X[boundary_sv_neg_idx, 0], X[boundary_sv_neg_idx, 1], c='blue', marker='D', 
                  s=30, label='Boundary SV', alpha=0.7)
    if np.any(boundary_sv_pos_idx):
        ax.scatter(X[boundary_sv_pos_idx, 0], X[boundary_sv_pos_idx, 1], c='red', marker='D', 
                  s=30, alpha=0.7)
    
    margin_sv_neg_idx = margin_sv_idx & (y == -1)
    margin_sv_pos_idx = margin_sv_idx & (y == 1)
    if np.any(margin_sv_neg_idx):
        ax.scatter(X[margin_sv_neg_idx, 0], X[margin_sv_neg_idx, 1], c='blue', marker='*', 
                  s=40, label='Margin SV', alpha=0.7)
    if np.any(margin_sv_pos_idx):
        ax.scatter(X[margin_sv_pos_idx, 0], X[margin_sv_pos_idx, 1], c='red', marker='*', 
                  s=40, alpha=0.7)
    
    ax.set_title(f'{title} (σ={sigma})')
    ax.set_xlabel('$t_1$')
    ax.set_ylabel('$t_2$')
    ax.legend(loc='upper right', fontsize='small')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    
    if show_plot:
        plt.tight_layout()
        plt.show()
    
    # Print counts if requested
    if print_stats:
        print(f"Number of support vectors for C={C}, σ={sigma}: {np.sum(support_idx)}")
        print(f"Number of margin defining vectors for C={C}, σ={sigma}: {np.sum(margin_sv_idx)}")
    
    return ax

def plot_svm_transformed_results(X_original, y, beta, b, C, title='SVM with Transformed Features', ax=None, figsize=(7, 5), print_stats=True):
    """
    Plots the results of SVM classification with the phi transformation in the original 2D space.
    
    Parameters:
    -----------
    X_original : array, shape (n_samples, 2)
        Original 2D input data before transformation
    y : array, shape (n_samples,)
        Target labels (-1 or 1)
    beta : array, shape (n_samples,)
        Optimized Lagrange multipliers from SVM training on transformed data
    b : float
        Optimized bias term from SVM training
    C : float
        Regularization parameter for SVM
    title : str, optional
        Title for the plot
    ax : matplotlib.axes.Axes, optional
        If provided, plot on this axis instead of creating a new figure
    figsize : tuple, optional
        Figure size (width, height) in inches, only used if ax is None
    print_stats : bool, optional
        Whether to print support vector statistics
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_plot = True
    else:
        show_plot = False
    
    margin = 0.5
    x_min, x_max = X_original[:, 0].min() - margin, X_original[:, 0].max() + margin
    y_min, y_max = X_original[:, 1].min() - margin, X_original[:, 1].max() + margin
    
    h = 0.02  
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    mesh_points_2d = np.c_[xx.ravel(), yy.ravel()]
    
    mesh_points_transformed = phi_transform(mesh_points_2d)
    
    Z = np.zeros(mesh_points_transformed.shape[0])
    
    X_transformed = phi_transform(X_original)
    
    for i in range(len(X_transformed)):
        if beta[i] > 0: 
            Z += beta[i] * y[i] * np.dot(X_transformed[i], mesh_points_transformed.T)
    Z += b
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=[-float('inf'), 0, float('inf')], 
                colors=['skyblue', 'salmon'], alpha=0.4)
    
    cs = ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    ax.clabel(cs, inline=1, fontsize=10, fmt='f = 0')
    
    support_idx = beta > 0
    
    regular_neg_idx = (y == -1) & ~support_idx
    regular_pos_idx = (y == 1) & ~support_idx
    
    boundary_sv_idx = (beta > 0) & (np.isclose(beta, C) | (beta >= C))
    margin_sv_idx = (beta > 0) & (beta < C) & ~np.isclose(beta, C)
    
    ax.scatter(X_original[regular_neg_idx, 0], X_original[regular_neg_idx, 1], c='blue', marker='o', 
              label='Class -1', s=20, alpha=0.7)
    ax.scatter(X_original[regular_pos_idx, 0], X_original[regular_pos_idx, 1], c='red', marker='o', 
              label='Class 1', s=20, alpha=0.7)
    
    boundary_sv_neg_idx = boundary_sv_idx & (y == -1)
    boundary_sv_pos_idx = boundary_sv_idx & (y == 1)
    if np.any(boundary_sv_neg_idx):
        ax.scatter(X_original[boundary_sv_neg_idx, 0], X_original[boundary_sv_neg_idx, 1], c='blue', marker='D', 
                  s=30, label='Boundary SV', alpha=0.7)
    if np.any(boundary_sv_pos_idx):
        ax.scatter(X_original[boundary_sv_pos_idx, 0], X_original[boundary_sv_pos_idx, 1], c='red', marker='D', 
                  s=30, alpha=0.7)
    
    margin_sv_neg_idx = margin_sv_idx & (y == -1)
    margin_sv_pos_idx = margin_sv_idx & (y == 1)
    if np.any(margin_sv_neg_idx):
        ax.scatter(X_original[margin_sv_neg_idx, 0], X_original[margin_sv_neg_idx, 1], c='blue', marker='*', 
                  s=40, label='Margin SV', alpha=0.7)
    if np.any(margin_sv_pos_idx):
        ax.scatter(X_original[margin_sv_pos_idx, 0], X_original[margin_sv_pos_idx, 1], c='red', marker='*', 
                  s=40, alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel('$t_1$')
    ax.set_ylabel('$t_2$')
    ax.legend(loc='upper right', fontsize='small')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    
    if show_plot:
        plt.tight_layout()
        plt.show()
    
    if print_stats:
        print(f"Number of support vectors for C={C}: {np.sum(support_idx)}")
        print(f"Number of margin defining vectors for C={C}: {np.sum(margin_sv_idx)}")
    
    return ax

def plot_lls_results(X, y, alpha, title='LLS Classifier', ax=None, figsize=(7, 5), print_coef=True):
    """
    Plots the results of Linear Least Squares classification.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
        Input data
    y : array, shape (n_samples,)
        Target labels (-1 or 1)
    alpha : array, shape (n_features + 1,)
        Coefficients from the linear least squares model
        [alpha_0, alpha_1, alpha_2, ...] where alpha_0 is the bias term
    title : str, optional
        Title for the plot
    ax : matplotlib.axes.Axes, optional
        If provided, plot on this axis instead of creating a new figure
    figsize : tuple, optional
        Figure size (width, height) in inches, only used if ax is None
    print_coef : bool, optional
        Whether to print coefficients (default: True)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_plot = True
    else:
        show_plot = False
    
    ax.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', label='Class -1', alpha=0.7)
    ax.scatter(X[y==1, 0], X[y==1, 1], color='red', label='Class 1', alpha=0.7)
    
    margin = 1.5 
    x1_min, x1_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    x2_min, x2_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                           np.linspace(x2_min, x2_max, 300))
    Z = alpha[0] + alpha[1] * xx1 + alpha[2] * xx2
    
    ax.contour(xx1, xx2, Z, levels=[0], colors='k', linestyles='-', linewidths=2)
    ax.contourf(xx1, xx2, Z, levels=[-float('inf'), 0, float('inf')], 
                colors=['skyblue', 'salmon'], alpha=0.4)
    
    ax.set_xlabel('$t_1$')
    ax.set_ylabel('$t_2$')
    ax.set_title(f'{title}: $\\alpha_0 + \\alpha_1 \\cdot t_1 + \\alpha_2 \\cdot t_2 = 0$')
    ax.legend(title="Classes")
    ax.grid(True, alpha=0.3)
    
    if show_plot:
        plt.tight_layout()
        plt.show()
    
    if print_coef:
        print(f"Coefficients: α₀ = {alpha[0]:.4f}, α₁ = {alpha[1]:.4f}, α₂ = {alpha[2]:.4f}")
    
    return ax