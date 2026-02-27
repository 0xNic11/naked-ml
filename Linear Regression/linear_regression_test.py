import numpy as np   
import matplotlib.pyplot as plt    
from sklearn.model_selection import train_test_split
from sklearn import datasets
from linear_regression import LinearRegression


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#print(X_train.shape)
#print(y_train.shape)

#plt.figure(figsize=(10,8))
#plt.scatter(X[: , 0], y, color ='r', marker='o', s=30)
#plt.show()

regressor = LinearRegression(lr=0.1)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = regressor.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

accuracy = regressor.r2_score(y_test, y_pred)
print(f"R2 Score: {accuracy}")


y_pred_line = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(10,8))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction Line")
plt.show()