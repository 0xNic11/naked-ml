import numpy as np    

class SupportVectorMachine:
    """
    A simple implementation of a Support Vector Machine for binary classification.
    """
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
        
    def fit(self, X, y):
        """
        Fit the Support Vector Machine to the training data.
        """
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0 
        
        self.loss_history = []
        
        for _ in range(self.n_iters):
            epoch_loss = 0
            for idx, x_i in enumerate(X):
                # Calculate the condition for the hinge loss
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_[idx] * x_i)
                    self.b -= self.lr * y_[idx]
                    epoch_loss += 1 - y_[idx] * (np.dot(x_i, self.w) - self.b)
            self.loss_history.append(epoch_loss)
            # Check for convergence
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-5:
                    print(f"Converged at iteration {_}")
                    break
    
    def predict(self, X):
        """
        Predict the class labels for the given data.
        """
        linear_output = np.dot(X, self.w) - self.b
        # Apply the sign function to get the predicted class labels
        predicted = np.sign(linear_output)
        return predicted