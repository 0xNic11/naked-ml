import numpy as np      
import matplotlib.pyplot as plt              

from sklearn.model_selection import train_test_split
from sklearn import datasets

from naive_bayes import NaiveBayes


X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
acc = nb.accuracy(y_test, y_pred)

print(f"Naive Bayes accuracy: {acc:.2f}")