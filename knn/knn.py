import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between two vectors.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    """
    K-Nearest Neighbors classifier.
    """
    
    def __init__(self, k=3):
        self.k = k
        
        
    def fit(self, X, y):
        """
        Fit the KNN model to the training data.
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """
        Predict the class labels for the given test data.
        """
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        """ 
        Predict the class label for a single test sample.
        """
        # compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train] 
        
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # majority vote, most common class label 
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def accuracy(self, y_pred, y_test):
        """
        Calculate the accuracy of the predictions.
        """
        return np.sum(y_pred == y_test) / len(y_test)
    
    
    