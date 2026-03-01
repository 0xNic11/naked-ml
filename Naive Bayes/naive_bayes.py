import numpy as np

class NaiveBayes:
    """
    A simple implementation of the Naive Bayes classifier.
    """
    
    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        """
        n_samples, n_features = X.shape
        self.unique_classes = np.unique(y)
        n_classes = len(self.unique_classes)
        
        # init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        
        
        for c in self.unique_classes:
            X_c = X[c == y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)
            
        
    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self, x):
        """
        Predict the class label for a single sample x.
        """
        posteriors = []
            
        for idx, c in enumerate(self.unique_classes):
            prior = np.log(self._priors[idx])
            cls_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + cls_conditional
            posteriors.append(posterior)
            
        return self.unique_classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        """
        Compute the probability density function for a given class and sample.
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx] + 1e-9 # add small value to avoid division by zero
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return np.clip(numerator / denominator, 1e-300, None) # clip to avoid underflow
    
    def accuracy(self, y, y_pred):
        """
        Compute the accuracy of predictions.
        """
        return np.sum(y == y_pred) / len(y)