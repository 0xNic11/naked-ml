import numpy as np 
import matplotlib.pyplot as plt            
from matplotlib.colors import ListedColormap
from sklearn import datasets         
from sklearn.model_selection import train_test_split
from knn import KNN 

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# importing iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(X_train.shape)
#print(X_train[0])
#print(y_train.shape)
#print(y_train)

#plt.figure(figsize=(10,10))
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
#plt.show()

clf = KNN(k=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


accuracy = clf.accuracy(y_pred, y_test)
print(accuracy)