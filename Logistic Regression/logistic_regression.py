import numpy as np          


class LogisticRegression:
    """
    Logistic Regression implementation from scratch
    """
    def __init__(self, lr=0.001, n_iters=1000, threshold=0.5):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.threshold = threshold
        
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)  # prevent overflow
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data
        """
        n_samples, n_features = X.shape
        
        # paramters init
        self.w = np.zeros(n_features)
        self.b = 0
        
        # gradient descent
        for _ in range(self.n_iters):
            # linear model
            linear_model = np.dot(X, self.w) + self.b
            
            # apply sigmoid function
            y_pred = self._sigmoid(linear_model)
            
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            # update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # check for convergence
            if np.linalg.norm(dw) < 1e-4:
                break
    
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        linear_model = np.dot(X, self.w) + self.b
        
        y_pred = self._sigmoid(linear_model)
        # convert probabilities to class labels
        y_pred_cls = (y_pred >= self.threshold).astype(int)
        
        return y_pred_cls
    
    def accuracy(self, y, y_pred):
        """
        Compute the accuracy of the model
        """
        accuracy = np.sum(y == y_pred) / len(y)
        return accuracy    