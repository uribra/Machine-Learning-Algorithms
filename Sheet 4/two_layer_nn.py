import numpy as np

class TwoLayerNN:
    """
    A two-layer (fully-connected, feed-forward) neural network.
    Architecture: Input -> Hidden (ReLU) -> Output (Identity)
    """
    
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the neural network.
        
        Args:
            input_dim (int): Dimension of input (d)
            hidden_dim (int): Number of hidden neurons (d2)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights and biases uniformly in (-1, 1)
        # W^(1): input -> hidden (shape: input_dim x hidden_dim)
        self.W1 = np.random.uniform(-1, 1, (input_dim, hidden_dim))
        
        # W^(2): hidden -> output (shape: hidden_dim x 1)
        self.W2 = np.random.uniform(-1, 1, (hidden_dim, 1))
        
        # b^(2): hidden layer bias (shape: hidden_dim,)
        self.b2 = np.random.uniform(-1, 1, hidden_dim)
        
        # b^(3): output layer bias (scalar)
        self.b3 = np.random.uniform(-1, 1)
        
        self.cache = {}
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation function."""
        return (x > 0).astype(float)
    
    def feedForward(self, X):
        """
        Forward propagation for a minibatch.
        
        Args:
            X (np.ndarray): Input data of shape (batch_size, input_dim)
            
        Returns:
            np.ndarray: Output predictions of shape (batch_size, 1)
        """
        batch_size = X.shape[0]
        
        # net^(2) = X @ W^(1) + b^(2)
        net2 = X @ self.W1 + self.b2  # Shape: (batch_size, hidden_dim)
        
        # o^(2) = ReLU(net^(2))
        o2 = self.relu(net2)  # Shape: (batch_size, hidden_dim)
        
        # net^(3) = o^(2) @ W^(2) + b^(3)
        net3 = o2 @ self.W2 + self.b3  # Shape: (batch_size, 1)
        
        # o^(3) = Identity(net^(3)) = net^(3)
        output = net3  # Shape: (batch_size, 1)
        
        self.cache = {
            'X': X,
            'net2': net2,
            'o2': o2,
            'net3': net3,
            'output': output
        }
        
        return output
    
    def backprop(self, y_true):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true (np.ndarray): True labels of shape (batch_size, 1)
            
        Returns:
            dict: Gradients for all parameters
        """
        batch_size = y_true.shape[0]
        
        X = self.cache['X']
        net2 = self.cache['net2']
        o2 = self.cache['o2']
        net3 = self.cache['net3']
        output = self.cache['output']
        
        # δ^(2) = 2(f(x) - y) = 2(output - y_true)
        delta2 = 2 * (output - y_true)  # Shape: (batch_size, 1)
        
        # δ^(1) = δ^(2) * φ'^(3)(net^(3)) * W^(2)
        # φ^(3) = identity, φ'^(3) = 1 =>
        # δ^(1) = δ^(2) @ W^(2).T ⊙ φ'^(2)(net^(2))
        delta1_before_activation = delta2 @ self.W2.T  # Shape: (batch_size, hidden_dim)
        phi2_derivative = self.relu_derivative(net2)  # Shape: (batch_size, hidden_dim)
        delta1 = delta1_before_activation * phi2_derivative
        
        # Compute gradients for W^(2) and b^(3)
        # ∇W^(2) = o^(2).T @ (δ^(2) ⊙ φ'^(3)(net^(3)))
        # Since φ'^(3) = 1: ∇W^(2) = o^(2).T @ δ^(2)
        grad_W2 = o2.T @ delta2 / batch_size  # Shape: (hidden_dim, 1)
        
        # ∇b^(3) = mean(δ^(2) ⊙ φ'^(3)(net^(3)))
        #φ'^(3) = 1: ∇b^(3) = mean(δ^(2))
        grad_b3 = np.mean(delta2)
        
        # ∇W^(1) = X.T @ (δ^(1) ⊙ φ'^(2)(net^(2)))
        grad_W1 = X.T @ delta1 / batch_size  # Shape: (input_dim, hidden_dim)
        
        # ∇b^(2) = mean(δ^(1) ⊙ φ'^(2)(net^(2)))
        grad_b2 = np.mean(delta1, axis=0)  # Shape: (hidden_dim,)
        
        return {
            'grad_W1': grad_W1,
            'grad_W2': grad_W2,
            'grad_b2': grad_b2,
            'grad_b3': grad_b3
        }
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute the mean squared error loss.
        
        Args:
            y_pred (np.ndarray): Predicted values
            y_true (np.ndarray): True values
            
        Returns:
            float: Mean squared error
        """
        return np.mean((y_pred - y_true) ** 2)
    
    def sample_minibatch(self, X, y, batch_size):
        """
        Randomly sample a minibatch from the dataset.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_dim)
            y (np.ndarray): Target labels of shape (n_samples, 1)
            batch_size (int): Size of the minibatch (K)
            
        Returns:
            tuple: (X_batch, y_batch) - randomly sampled minibatch
        """
        n_samples = X.shape[0]
        
        batch_indices = np.random.choice(n_samples, size=batch_size, replace=False)
        
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        return X_batch, y_batch
    
    def train_sgd(self, X, y, learning_rate, batch_size, num_steps, verbose=False):
        """
        Train the network using stochastic minibatch gradient descent.
        Implements Algorithm 4.4 from the document.
        
        Args:
            X (np.ndarray): Training input data of shape (n_samples, input_dim)
            y (np.ndarray): Training labels of shape (n_samples, 1)
            learning_rate (float): Learning rate ν > 0
            batch_size (int): Minibatch size K
            num_steps (int): Number of training steps S
            verbose (bool): Whether to print training progress
            
        Returns:
            list: History of losses during training
        """
        n_samples = X.shape[0]
        loss_history = []
        
        for step in range(num_steps):
            X_batch, y_batch = self.sample_minibatch(X, y, batch_size)
            
            _ = self.feedForward(X_batch) #necessary to cache the correct outputs. maybe not best solution to be honest but works
            
            gradients = self.backprop(y_batch)
            
            self.W1 -= learning_rate * gradients['grad_W1']
            self.W2 -= learning_rate * gradients['grad_W2']
            self.b2 -= learning_rate * gradients['grad_b2']
            self.b3 -= learning_rate * gradients['grad_b3']
            
            if step % 100 == 0 or step == num_steps - 1:
                y_pred_full = self.feedForward(X)
                loss = self.compute_loss(y_pred_full, y)
                loss_history.append(loss)
                
                if verbose:
                    print(f"Step {step:6d}/{num_steps}: Loss = {loss:.6f}")
        
        return loss_history