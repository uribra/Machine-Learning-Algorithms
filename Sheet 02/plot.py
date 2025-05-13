# Import standard modules
import numpy as np
import random 
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------#
def scatter_hyper_plot(X, y, C, f, beta, b):

    x1_max = max(X[:,0])
    x1_min = min(X[:,0])
    x2_max = max(X[:,1])
    x2_min = min(X[:,1])
    
    x1_vals, x2_vals = np.meshgrid(np.linspace(x1_min -1 , x1_max +1, 300),
                       np.linspace(x2_min-1, x2_max+1, 300))
    

    grid_points = np.c_[x1_vals.ravel(), x2_vals.ravel()]  

    # Evaluate f on each grid point
    Z_flat = np.apply_along_axis(lambda t: f(t, X, y, beta, b, kernel='gauss_kernel'), 1, grid_points)
    Z_trans = Z_flat.reshape(x1_vals.shape)

    indices_negative = np.where(y=-1)[0]
    indices_positive = np.where(y=1)[0]

    # Obtain the support vectors beta_k > 0 and margin vectors 
    support_indices = np.where(beta > 0)[0]
    margin_indices = np.where((C > beta)  & (beta >0))[0]
    support_vectors = X[support_indices]
    margin_vectors = X[margin_indices]
    # Extract the support and margin vectors
    support_vectors = X[support_indices]
    margin_vectors = X[margin_indices]
    

    # Scatter plot with separating hyperplane
    plt.figure()
    plt.scatter(X[indices_negative,0], X[indices_negative,1], color = "blue", label = "Class -1")
    plt.scatter(X[indices_positive,0], X[indices_positive,1], color = "red", label = "Class 1")
    plt.scatter(support_vectors[:,0], support_vectors[:,1], marker='*' , color = 'black', label = 'Support vector')
    plt.scatter(margin_vectors[:,0], margin_vectors[:,0], marker='+', color = 'yellow', label = 'Margin vector')
    plt.contour(x1_vals, x2_vals, Z_trans, levels = [0], colors='k', linestyles='-', linewidths=2)
    plt.contourf(x1_vals, x2_vals, Z_trans, levels=[-np.inf, 0, np.inf], colors=['lightblue', 'lightcoral'], alpha=0.5)

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('SVM Classification C=%.2f' %C)
    plt.legend()
    plt.show()
    plt.figure()   
#-----------------------------------------------------------------------------------------#
