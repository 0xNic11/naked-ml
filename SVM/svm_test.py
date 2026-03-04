import numpy as np          
import matplotlib.pyplot as plt              
from sklearn import datasets

from svm import SupportVectorMachine

X,y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=42)


svm = SupportVectorMachine(lr=0.01)
svm.fit(X,y)


print(svm.w, svm.b)


def vis_svm():
    """
    Visualize the Support Vector Machine.
    """
    def get_hyperplane_value(x, w, b, offset):
        """
        Calculate the value of the hyperplane at a given x.
        """
        return (-w[0] * x + b + offset) / w[1]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y) 
    
    # Get the minimum and maximum x values to plot the hyperplane
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])
    
    # Get the corresponding x1 values for the hyperplane and the margins
    x1_1 = get_hyperplane_value(x0_1, svm.w, svm.b, 0)
    x1_2 = get_hyperplane_value(x0_2, svm.w, svm.b, 0)
    
    x1_1_m = get_hyperplane_value(x0_1, svm.w, svm.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, svm.w, svm.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, svm.w, svm.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, svm.w, svm.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()

vis_svm()