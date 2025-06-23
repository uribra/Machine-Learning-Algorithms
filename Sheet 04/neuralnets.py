import numpy as np
import random 
class TwoLayerNNParameters: 
    def __init__(self, dims, nu, S, K):
        self.dims = dims            # Dimension of initial, hidden and output layer
        self.nu = nu                # Learning rate for stochastic gradient descent
        self.S = S                  # Number of iterations in the stochastic gradient descent
        self.K = K                  # Batch size

    def __repr__(self):
        return (f"TwoLayerNNParameters:\n"
                f"  Dims: {self.dims}\n"
                f"  Learning rate (nu): {self.nu}\n"
                f"  Iterations (S): {self.S}\n"
                f"  Batch size (K): {self.K}")


class TwoLayerNN:

    """
    A class representing a two-layer fully connected neural network, 
    trained using Stochastic Gradient Descent and ReLU activation.

    The network consists of:
        - An input layer with dimension d1
        - A hidden layer with dimension d2 and ReLU activation
        - An output layer with dimension d3 and linear activation

    Parameters
    ----------
    nu : float
        Learning rate used in the stochastic gradient descent.
    K : int
        Mini-batch size used during training.
    dims : list of int
        A list specifying the number of neurons in each layer: [input_dim, hidden_dim, output_dim].
    S : int
        Number of training iterations (steps of SGD).

    Attributes
    ----------
    phi : list of callable
        Activation functions used in the network; ReLU for hidden layer, identity for output.
    phi_prime : list of callable
        Derivatives of activation functions for use in backpropagation.

    Methods
    -------
    feedForward(W, b, z):
        Computes forward pass for a single input example.
    
    feedForward_batch(W, b, Z):
        Computes forward pass for a batch of input examples.
    
    backpropagation(W, b, z, y):
        Computes gradients for a single training example via backpropagation.
    
    backpropagation_batch(W, b, Z):
        Placeholder for batch backpropagation (not yet implemented).
    
    sample_minibatch(Z, y, K):
        Randomly samples a mini-batch of size K from the dataset.

    least_squares_error(y_pred, y_true):
        Computes mean squared error between predictions and targets.
    
    train():
        Trains the neural network using SGD on the provided data.
    
    predict(W, b, Z):
        Performs prediction using trained weights and biases on new input data.
    """

    
    def __init__(self, parameters: TwoLayerNNParameters):
        
        self.dims = parameters.dims
        self.nu = parameters.nu
        self.K = parameters.K 
        self.S = parameters.S

        # Define the activation functions and derivatives for the two layer NN
        def phi_2(z):
            return np.maximum(z, 0)
        def phi_3(z):
            return z
        def phi_2_prime(z):
            return (z >= 0).astype(int)
        def phi_3_prime(z):
            return 1

        # Collect activation functions phi_i into a list
        self.phi = [phi_2, phi_3]
        self.phi_prime = [phi_2_prime, phi_3_prime]


    def feedForward(self, W, b, z):
        """
        Computes the feed forward for a two layer neural network
    
        Parameters
        ----------
        W: (2, d_1, d_2) NumPy array, 
            Each W[l] is a weight matrix for layer l of shape (d_{l+1}, d_l)
        b: (2, d_1, d_2) NumPy array
            Each b[l] is a bias vector for layer l of shape (d_{l+1},)
        phi: List of functions
            Activation functions for each layer l=1,2
        z: (d, ) NumPy array
            Input vector of shape (d_0,)

        Returns
        -------
        o : NumPy array
            Output of the final layer (after applying all weights, biases, activations)
    
        """
        phi = self.phi

        # Initialize the values for the Input layer
        o = z
        net_vals = [None]*2
        o_vals = [None]*3
        o_vals[0] = o

        for l in range(2):
            net = np.dot(W[l].T, o) + b[l]
            net_vals[l]= net
            # Apply activation function to net (Understood component wise)
            o_new = phi[l](net)
            o = o_new
            o_vals[l+1] = o
    
        return o, net_vals, o_vals


    def feedForward_batch(self, W, b, Z):
        """
        Computes the feed forward for a input batch in a two layer neural network
    
        Parameters
        ----------
        W: List of NumPy arrays, 
            W[0] is a weight matrix for layer l=1 of shape (d1, d2), W[1] is a weight matrix for layer l=2 of shape (d2, d),
        b: List of NumPy arrays
            Each b[l] is a bias vector for layer l of shape (d_{l},)
        phi: List of functions
            Activation functions for the hidden layer and output layer
        Z: (K,d1) NumPy array
            Input vector of shape (d1,)

        Returns
        -------
        o : NumPy array
            Output of the final layer after applying all weights, biases and activations
        net_vals: list 
            List of NumPy arrays
        o_vals: list
            List of NumPy arrays 

        """
        #o = Z
        #net_vals = [None]*2
        #o_vals = [None]*3
        #o_vals[0] = o
        #for l in range(2):
        #    net =  o @ W[l] + b[l]
        #    net_vals[l]= net
        #    o_new = phi[l](net)
        #    o = o_new
        #    o_vals[l+1] = o

        assert W[0].shape == (self.dims[0], self.dims[1])
        assert W[1].shape == (self.dims[1], self.dims[2])
        assert b[0].shape == (self.dims[1],)
        assert b[1].shape == (self.dims[2],)
        assert Z.shape[1] == self.dims[0]

        o1 = Z
        net2 = o1 @ W[0] + b[0]     # Shape: (batch_size, d2)
        o2 = self.phi[0](net2)      # Shape: (batch_size, d2)
        net3 = o2 @ W[1] + b[1]     # Shape: (batch_size, 1)
        o3 = net3                   # Shape: (batch_size, 1)
        output = o3.copy()
        o_vals = [o1, o2, o3]
        net_vals = [net2, net3]
    
        return output, net_vals, o_vals

    def backpropagation_batch(self, W, b, Z, y):
        '''
        Parameters
        ----------
        W: list of weights (W[0]: (d1, d2), W[1]: (d2, d3))
        b: list of biases (b[0]: (d2,), b[1]: (d3,))
        Z_batch: (N, d1) input batch
        Y_batch: (N, d3) output batch

        Returns
        -------
        grad_W: list of gradients for weights
        grad_b: list of gradients for biases
        '''
        n_obs = Z.shape[0]

        f_z, net_vals, o_vals = self.feedForward_batch(W, b, Z)
        # Compute deltas
        delta2 = 2 * (f_z - y.reshape((n_obs,1)))          # Shape: (N, d3)
        delta1 = (delta2 @ W[1].T)      # Shape: (N, d2)

        net2 = net_vals[0]
        o2 = o_vals[1]

        # Compute gradients
        delta_1_active = (delta1*((net2 >= 0)))
        grad_W2 = (o2.T @ delta2) / n_obs                       # Shape: (d2, d3)
        grad_W1 = (Z.T @ delta_1_active) / n_obs                # Shape: (d1, d2)

        grad_b3 = np.mean(delta2, axis=0)                       # Shape: (d3,)            
        grad_b2 = np.mean(delta_1_active, axis=0)               # Shape: (d2,)

        grad_W = [grad_W1, grad_W2]
        grad_b = [grad_b2, grad_b3]

        return grad_W, grad_b


    def sample_minibatch(self, Z, y, K):
        '''
        Function to randomly sample a batch of size K from a data set

        Parameters
        ----------
        X: (n_obs, n_features) NumPy array, 
            Data set of features
        y: (n_obs, ) NumPy array
            Data set of labels
        K: int
            Size of mini batch
    
        Returns
        -------
        X_batch: (K, n_features) NumPy array, 
        y_batch: (n_obs, ) NumPy array
        '''
        # Number of observations
        n_obs = Z.shape[0]
    
        # Draw the batch indices in the iteration s
        batch_indices = np.random.randint(0, n_obs, size = K)
        # Extract the data points for the batch from the data set X
        X_batch = Z[batch_indices]
        # Extract the labels for the batch from y
        y_batch = y[batch_indices]

        return X_batch, y_batch
    

    def least_squares_error(self, y_pred, y_true):
        ''''
        Function to compute the least squares error
        '''

        loss = np.mean((y_pred - y_true)**2)
        return loss 
    

    def initialize_parameters(self):
        """
        Initializes the weights and biases for a two-layer fully connected neural network.
        Weights are initialized uniformly in the range [-1, 1], and biases are initialized similarly.

        Returns
        -------
        W : list of np.ndarray
            List containing weight matrices:
            - W[0]: shape (d1, d2), weights from input to hidden layer
            - W[1]: shape (d2, d3), weights from hidden to output layer
        b : list of np.ndarray
            List containing bias vectors:
            - b[0]: shape (d2,), biases for hidden layer
            - b[1]: shape (d3,), biases for output layer
        """
        
        d1 = self.dims[0]
        d2 = self.dims[1]
        d3 = self.dims[2]

        # Initial layer
        W1 = np.random.uniform(-1, 1, (d1, d2))
        b2 = np.random.uniform(-1, 1, (d2,))
        # Hidden layer
        W2 =  np.random.uniform(-1, 1, (d2, d3))
        b3 = np.random.uniform(-1, 1, (d3,))

        # Store the initialized weight matrix and bias in arrays
        W = [W1, W2]
        b = [b2, b3]
        return W, b
    

    def train(self, X, y):

        '''
        Trains the 2 Layer Neural Net by Stochastic Gradient Descent

        Parameters
        ----------
        X: (n_obs, n_features) NumPy array, 
            Data set of features
        y: (n_obs, ) NumPy array
            Data set of labels
    
        Returns
        -------
        W: List of NumPy arrays, 
            W[0] is a weight matrix for layer l=1 of shape (d1, d2), 
            W[1] is a weight matrix for layer l=2 of shape (d2, d3),
        b: List of NumPy arrays
            b[0] is a bias vector for layer 2 of shape (d_{2},)
            b[1] is a bias vector for layer 3 of shape (d_{3},)
        '''

        nu = self.nu
        K = self.K
        S = self.S

        # Initialize the weight matrix and the bias b
        W, b = self.initialize_parameters()
    
        for s in range(S): 

            # Sample the mini batch of size K
            X_batch, y_batch = self.sample_minibatch(X, y, K)
            grad_W_C, grad_b_C =  self.backpropagation_batch(W, b, X_batch, y_batch)

            for l in range(0,2):
                # Update the weight matrix W and bias b for layer l
                W[l] -= nu*grad_W_C[l]
                b[l] -=  nu*grad_b_C[l]

            if s % 5000 == 0:
                y_pred_batch = self.predict(W, b, X_batch)
                print(f"Least Squares Error after {s} iterations: {self.least_squares_error(y_pred_batch, y_batch)}")
        # Store the trained weights and bias        
        return W, b
    
    def predict(self, W, b, Z): 

        '''
        Parameters
        ----------
        W: List of NumPy arrays
            Each W[l] is a weight matrix for layer l of shape (d_{l}, d_{l+1})
        b: List of NumPy arrays
            Each b[l] is a bias vector for layer l of shape (d_{l},)
        Z: (n_obs, n_features) NumPy array, 
            Data set of features
        
        Returns
        -------
        y_predict: NumPy array, shape (n_obs,)
            Predicted output values 
        '''
        y_predicted = self.feedForward_batch(W, b, Z)[0]
        return y_predicted