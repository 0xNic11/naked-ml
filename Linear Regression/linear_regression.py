import numpy as np          

class LinearRegression:
    
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    
    def fit(self, X, y):
        
        # parameters init
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        # gradient descent
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.w) + self.b 
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def predict(self, X):
        y_pred = np.dot(X, self.w) + self.b
        return y_pred
    
    def mean_squared_error(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)
    
    def r2_score(self, y, y_pred):
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        if ss_total == 0:
            return 1.0 if ss_residual == 0 else 0.0
        r2 = 1 - (ss_residual / ss_total)
        return r2