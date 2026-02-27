import numpy as np  
import matplotlib.pyplot as plt                   

from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic_regression import LogisticRegression

breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LogisticRegression(lr=0.1, threshold=0.5)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

accuracy = regressor.accuracy(y_test, predictions)
print("Logistic Regression classification accuracy:", accuracy)