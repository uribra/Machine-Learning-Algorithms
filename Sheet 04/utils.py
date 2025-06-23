import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_predicted_label, y_true):

    """
        Confusion matrix for classification

        *Args: 
            y_predicted_label: Predicted labels data, shape (n_samples,)
            y_true: Target labels, shape (n_samples,)
        Returns: 
            C: Confusion matrix, shape (2,2)
    """
    n_obs_test = y_predicted_label.shape[0]
        # Generate the 2x2 confusion matrix
    C = np.zeros((2, 2), dtype = int)

    for i in range(n_obs_test): 
        if (y_predicted_label[i] == -1 and y_true[i] == -1):
            C[0,0] +=1
        if (y_predicted_label[i] == -1 and y_true[i] == 1):
            C[0,1] +=1
        if (y_predicted_label[i] == 1 and y_true[i] == -1):
            C[1,0] +=1
        if (y_predicted_label[i] == 1 and y_true[i] == 1):
            C[1,1] +=1

    return C


def accuracy(y_predicted, y_true):

    """
        Accuracy for from confusion matrix for classification with +1/-1 labels

        *Args: 
            C: Confusion matrix, shape (n_features, n_features)
        Returns: 
            accuracy, float
    """
    y_predicted_label = np.sign(y_predicted)
    C = confusion_matrix(y_predicted_label, y_true)
    return np.trace(C)/np.sum(C)


def plot_decision_boundary(nn, W, b, X, y, title = None):

    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    x1_vals, x2_vals = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
    mesh = np.c_[x1_vals.ravel(), x2_vals.ravel()]  

    # Evaluate f on each grid point
    Z = np.apply_along_axis(lambda t: nn.feedForward(W, b, t)[0], 1, mesh)
    Z = Z.reshape(x1_vals.shape)

    indices_pos = np.where(y==1)
    indices_neg = np.where(y==-1)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[indices_neg, 0], X[indices_neg, 1], color = "blue", label = "Class -1")
    plt.scatter(X[indices_pos, 0], X[indices_pos, 1], color = "red", label = "Class 1")
    plt.contour(x1_vals, x2_vals, Z, levels = [0], colors='k', linestyles='-', linewidths=2)
    plt.contourf(x1_vals, x2_vals, Z, levels=[-np.inf, 0, np.inf], colors=['lightblue', 'lightcoral'], alpha=0.5)

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.legend()
    plt.show()


def scatterplot_features(X,y, title = None):

    indices_pos = np.where(y==1)
    indices_neg = np.where(y==-1)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[indices_neg,0], X[indices_neg,1], color = "blue", label = "Class -1")
    plt.scatter(X[indices_pos,0], X[indices_pos,1], color = "red", label = "Class 1")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.legend()
    plt.show()



def generate_circle_data(n_positive, n_negative):
    draws_1 = []
    while len(draws_1) < n_negative:
        unif_draw = list(np.random.uniform([-1, -1], [1, 1], (2,)))
        if np.linalg.norm(unif_draw) <= 1:
            draws_1.append(unif_draw)
    draws_1 = np.array(draws_1)
                
    draws_2 = []
    while len(draws_2) < n_positive:
        unif_draw = list(np.random.uniform([-2, -2], [2, 2], (2,)))
        if np.linalg.norm(unif_draw) > 1 and np.linalg.norm(unif_draw) <= 2:
            draws_2.append(unif_draw)
    draws_2 = np.array(draws_2)

    # Stack the draws into numpy array
    X = np.vstack((draws_1, draws_2))
    # Generate the classified data
    y = np.hstack([(-1)*np.ones(n_negative), np.ones(n_positive)])

    return X, y

