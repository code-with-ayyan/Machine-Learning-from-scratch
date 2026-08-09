import numpy as np


class CustomSVR:
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', degree=3, coef0=0.0, tol=1e-3, max_iter=1000):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        
        self.alpha = None
        self.alpha_star = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.n_samples = None
        self.gamma_val = None
    
    def _kernel_function(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (self.gamma_val * np.dot(X1, X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            if X1.ndim == 1:
                X1 = X1.reshape(1, -1)
            if X2.ndim == 1:
                X2 = X2.reshape(1, -1)
            dist_sq = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma_val * dist_sq)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma_val * np.dot(X1, X2.T) + self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X):
        return self._kernel_function(X, X)
    
    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.float64).flatten()
        self.n_samples = self.X_train.shape[0]
        
        if self.gamma == 'scale':
            self.gamma_val = 1.0 / (self.X_train.shape[1] * np.var(self.X_train))
        elif self.gamma == 'auto':
            self.gamma_val = 1.0 / self.X_train.shape[1]
        else:
            self.gamma_val = float(self.gamma)
        
        K = self._compute_kernel_matrix(self.X_train)
        
        self.alpha = np.zeros(self.n_samples, dtype=np.float64)
        self.alpha_star = np.zeros(self.n_samples, dtype=np.float64)
        
        for iteration in range(self.max_iter):
            alpha_prev = self.alpha.copy()
            alpha_star_prev = self.alpha_star.copy()
            
            for i in range(self.n_samples):
                prediction = self._predict_single(i, K)
                error = prediction - self.y_train[i]
                
                if error > self.epsilon + self.tol:
                    self.alpha_star[i] = min(self.alpha_star[i] + error, self.C)
                elif error < -self.epsilon - self.tol:
                    self.alpha[i] = min(self.alpha[i] - error, self.C)
                
                self.b = self._update_bias(i, K)
            
            diff = np.max(np.abs(self.alpha - alpha_prev)) + np.max(np.abs(self.alpha_star - alpha_star_prev))
            if diff < self.tol:
                break
        
        sv_mask = (self.alpha > self.tol) | (self.alpha_star > self.tol)
        self.support_vectors = self.X_train[sv_mask]
        self.support_vector_labels = self.y_train[sv_mask]
        self.alpha = self.alpha[sv_mask]
        self.alpha_star = self.alpha_star[sv_mask]
        
        return self
    
    def _predict_single(self, i, K):
        diff = self.alpha - self.alpha_star
        return np.sum(diff * K[:, i]) + self.b
    
    def _update_bias(self, i, K):
        diff = self.alpha - self.alpha_star
        b_new = self.y_train[i] - np.sum(diff * K[:, i])
        return b_new
    
    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        K_test = self._kernel_function(X, self.X_train)
        diff = self.alpha - self.alpha_star
        predictions = np.dot(K_test, diff) + self.b
        
        return predictions
