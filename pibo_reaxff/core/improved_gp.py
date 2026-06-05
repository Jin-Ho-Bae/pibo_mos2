"""
Improved Gaussian Process with ARD and adaptive kernels for PIBO
"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


class ImprovedGaussianProcess:
    """Gaussian Process with ARD Matern kernel and stability improvements"""
    
    def __init__(self, kernel_type='matern', nu=2.5):
        self.kernel_type = kernel_type
        self.nu = nu  # For Matern kernel
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
        
        # Hyperparameters  
        self.length_scales = None  # ARD length scales
        self.signal_variance = 1.0
        self.noise_variance = 0.01
        
        # Normalization
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def kernel(self, X1, X2=None):
        """Compute kernel matrix with ARD"""
        if X2 is None:
            X2 = X1
            
        # Scale by ARD length scales
        if self.length_scales is not None:
            X1_scaled = X1 / self.length_scales
            X2_scaled = X2 / self.length_scales
        else:
            X1_scaled = X1
            X2_scaled = X2
            
        if self.kernel_type == 'rbf':
            # RBF kernel
            dists = cdist(X1_scaled, X2_scaled, 'sqeuclidean')
            K = self.signal_variance * np.exp(-0.5 * dists)
            
        elif self.kernel_type == 'matern':
            # Matern kernel
            dists = cdist(X1_scaled, X2_scaled, 'euclidean')
            
            if self.nu == 0.5:
                K = self.signal_variance * np.exp(-dists)
            elif self.nu == 1.5:
                K = self.signal_variance * (1 + np.sqrt(3) * dists) * np.exp(-np.sqrt(3) * dists)
            elif self.nu == 2.5:
                K = self.signal_variance * (1 + np.sqrt(5) * dists + 5/3 * dists**2) * np.exp(-np.sqrt(5) * dists)
            else:
                # Default to RBF
                K = self.signal_variance * np.exp(-0.5 * dists**2)
                
        else:
            # Default RBF
            dists = cdist(X1_scaled, X2_scaled, 'sqeuclidean')
            K = self.signal_variance * np.exp(-0.5 * dists)
            
        return K
    
    def fit(self, X, y, optimize_hyperparams=True):
        """Fit GP to training data"""
        # Normalize data
        self.X_train = self.scaler_X.fit_transform(X)
        self.y_train = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Initialize ARD length scales
        if self.length_scales is None:
            self.length_scales = np.ones(X.shape[1])
            
        if optimize_hyperparams:
            self.optimize_hyperparameters()
            
        # Compute kernel matrix
        K = self.kernel(self.X_train, self.X_train)
        K += self.noise_variance * np.eye(len(K))
        
        # Add jitter for numerical stability
        jitter = 1e-6
        K += jitter * np.eye(len(K))
        
        # Compute Cholesky decomposition
        try:
            L = cholesky(K, lower=True)
            self.L = L
            
            # Compute alpha = K^{-1} y
            self.alpha = solve_triangular(L.T, solve_triangular(L, self.y_train, lower=True))
            
        except np.linalg.LinAlgError:
            # Fallback to eigendecomposition for ill-conditioned matrices
            eigvals, eigvecs = np.linalg.eigh(K)
            eigvals = np.maximum(eigvals, 1e-10)
            K_inv = eigvecs @ np.diag(1/eigvals) @ eigvecs.T
            self.alpha = K_inv @ self.y_train
            
    def predict(self, X_test, return_std=True):
        """Predict mean and variance at test points"""
        X_test_norm = self.scaler_X.transform(X_test)
        
        # Compute kernel vectors
        K_star = self.kernel(self.X_train, X_test_norm)
        
        # Mean prediction
        mean = K_star.T @ self.alpha
        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        
        if not return_std:
            return mean
            
        # Variance prediction
        K_star_star = self.kernel(X_test_norm, X_test_norm)
        
        if hasattr(self, 'L'):
            v = solve_triangular(self.L, K_star, lower=True)
            var = np.diag(K_star_star) - np.sum(v**2, axis=0)
        else:
            # Fallback computation
            var = np.diag(K_star_star) - np.diag(K_star.T @ self.K_inv @ K_star)
            
        var = np.maximum(var, 1e-10)
        std = np.sqrt(var) * self.scaler_y.scale_[0]
        
        return mean, std
    
    def optimize_hyperparameters(self):
        """Simple hyperparameter optimization using marginal likelihood"""
        # This is a simplified version - in practice use scipy.optimize
        
        def neg_log_marginal_likelihood(params):
            self.signal_variance = np.exp(params[0])
            self.noise_variance = np.exp(params[1]) 
            self.length_scales = np.exp(params[2:])
            
            K = self.kernel(self.X_train, self.X_train)
            K += self.noise_variance * np.eye(len(K))
            
            try:
                L = cholesky(K, lower=True)
                alpha = solve_triangular(L.T, solve_triangular(L, self.y_train, lower=True))
                
                # Negative log marginal likelihood
                nll = 0.5 * self.y_train @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(self.y_train) * np.log(2*np.pi)
                return nll
                
            except:
                return 1e10
                
        # Simple grid search (replace with proper optimization)
        best_nll = np.inf
        best_params = None
        
        for signal_var in [0.5, 1.0, 2.0]:
            for noise_var in [0.001, 0.01, 0.1]:
                params = [np.log(signal_var), np.log(noise_var)] + [0.0] * self.X_train.shape[1]
                nll = neg_log_marginal_likelihood(params)
                
                if nll < best_nll:
                    best_nll = nll
                    best_params = params
                    
        if best_params is not None:
            self.signal_variance = np.exp(best_params[0])
            self.noise_variance = np.exp(best_params[1])
            self.length_scales = np.exp(best_params[2:])


class DeepKernelGP:
    """Deep Kernel Learning GP for highly nonlinear systems"""
    
    def __init__(self, input_dim, hidden_dims=[50, 30], kernel_type='rbf'):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_type = kernel_type
        
        # Neural network layers (simplified - use PyTorch in practice)
        self.weights = []
        self.biases = []
        
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0/dims[i])
            b = np.zeros(dims[i+1])
            self.weights.append(W)
            self.biases.append(b)
            
        # Base GP
        self.gp = ImprovedGaussianProcess(kernel_type=kernel_type)
        
    def transform(self, X):
        """Transform input through neural network"""
        H = X
        for W, b in zip(self.weights, self.biases):
            H = np.tanh(H @ W + b)  # Tanh activation
        return H
        
    def fit(self, X, y):
        """Fit deep kernel GP"""
        # Transform inputs
        X_transformed = self.transform(X)
        
        # Fit GP on transformed features
        self.gp.fit(X_transformed, y)
        
    def predict(self, X_test, return_std=True):
        """Predict using deep kernel GP"""
        X_test_transformed = self.transform(X_test)
        return self.gp.predict(X_test_transformed, return_std)
