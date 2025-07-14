import numpy as np
import matplotlib.pyplot as plt
from two_layer_nn import TwoLayerNN

def generate_circular_dataset(n_samples=500, random_state=42):
    """
    Generate exactly what's requested:
    - 250 points with ||x|| ≤ 1, labeled y = -1
    - 250 points with 1 < ||x|| ≤ 2, labeled y = 1
    
    Returns:
        X (np.ndarray): Shape (500, 2) - all points
        y (np.ndarray): Shape (500, 1) - labels (-1 or 1)
    """

    if n_samples % 2 == 1:
        n_samples -= 1

    n_samples_per_class = n_samples // 2

    np.random.seed(random_state)
    
    r1 = np.sqrt(np.random.uniform(0, 1, n_samples_per_class))
    theta1 = np.random.uniform(0, 2*np.pi, n_samples_per_class)
    
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    X_inner = np.column_stack([x1, y1])
    labels_inner = -np.ones((n_samples_per_class, 1))
    
    r2 = np.sqrt(np.random.uniform(1, 4, n_samples_per_class))
    theta2 = np.random.uniform(0, 2*np.pi, n_samples_per_class)
    
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    X_outer = np.column_stack([x2, y2])
    labels_outer = np.ones((n_samples_per_class, 1))
    
    X = np.vstack([X_inner, X_outer])
    y = np.vstack([labels_inner, labels_outer])
    
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y

def plot_dataset(X, y, title="Circular Classification Dataset"):
    """
    Plot the dataset with different colors for each class.
    Updated for new labeling scheme:
    - y = -1: points with ||x|| ≤ 1 (inner circle)
    - y = 1: points with 1 < ||x|| ≤ 2 (outer annulus)
    """
    plt.figure(figsize=(10, 8))
    
    inner_mask = (y.flatten() == -1)
    plt.scatter(X[inner_mask, 0], X[inner_mask, 1], 
               c='blue', alpha=0.6, label='Inner Circle (y=-1, ||x|| ≤ 1)', s=20)
    
    outer_mask = (y.flatten() == 1)
    plt.scatter(X[outer_mask, 0], X[outer_mask, 1], 
               c='red', alpha=0.6, label='Outer Annulus (y=1, 1 < ||x|| ≤ 2)', s=20)
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    circle1_x = 1 * np.cos(theta)
    circle1_y = 1 * np.sin(theta)
    plt.plot(circle1_x, circle1_y, 'k--', linewidth=2, label='Inner Boundary (r=1)')
    
    circle2_x = 2 * np.cos(theta)
    circle2_y = 2 * np.sin(theta)
    plt.plot(circle2_x, circle2_y, 'k:', linewidth=2, label='Outer Boundary (r=2)')
    
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    
    plt.show()

def plot_decision_boundary(nn, X, y, title="Neural Network Decision Boundary"):
    """
    Plot the decision boundary learned by the neural network.
    Updated for new labeling scheme:
    - y = -1: inner circle (||x|| ≤ 1) 
    - y = 1: outer annulus (1 < ||x|| ≤ 2)
    """
    plt.figure(figsize=(12, 5))
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.feedForward(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Network Output')
    
    inner_mask = (y.flatten() == -1)
    plt.scatter(X[inner_mask, 0], X[inner_mask, 1], c='blue', edgecolors='black', label='Inner Circle (y=-1)', s=30)
    
    outer_mask = (y.flatten() == 1)
    plt.scatter(X[outer_mask, 0], X[outer_mask, 1], c='red', edgecolors='black', label='Outer Annulus (y=1)', s=30)
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    circle1_x = 1 * np.cos(theta)
    circle1_y = 1 * np.sin(theta)
    plt.plot(circle1_x, circle1_y, 'k--', linewidth=3, label='Inner Boundary (r=1)')
    
    circle2_x = 2 * np.cos(theta)
    circle2_y = 2 * np.sin(theta)
    plt.plot(circle2_x, circle2_y, 'k:', linewidth=3, label='Outer Boundary (r=2)')
    
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Decision Boundary (Continuous)')
    plt.legend()
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    
    Z_binary = (Z > 0).astype(int) * 2 - 1
    
    plt.contourf(xx, yy, Z_binary, levels=[-1, 0, 1], colors=['blue', 'red'], alpha=0.3)

    plt.scatter(X[inner_mask, 0], X[inner_mask, 1], 
               c='blue', edgecolors='black', label='Inner Circle (y=-1)', s=30)
    plt.scatter(X[outer_mask, 0], X[outer_mask, 1], 
               c='red', edgecolors='black', label='Outer Annulus (y=1)', s=30)
    
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2, 
                linestyles='-', label='Learned Boundary')
    
    plt.plot(circle1_x, circle1_y, 'k--', linewidth=3, label='Inner Boundary (r=1)')
    plt.plot(circle2_x, circle2_y, 'k:', linewidth=3, label='Outer Boundary (r=2)')
    
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Binary Classification (Threshold=0)')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def compute_accuracy(y_pred, y_true, threshold=0):
    """Compute classification accuracy for -1/+1 labels."""
    # Convert predictions to -1/+1 format (not 0/1)
    y_pred_binary = np.where(y_pred > threshold, 1, -1)
    accuracy = np.mean(y_pred_binary.flatten() == y_true.flatten())
    return accuracy