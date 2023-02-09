# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 14:39:35 2018

@author: olive
"""

# =============================================================================
# Import modules
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adaline import AdalineGD
from matplotlib.colors import ListedColormap



# =============================================================================
# Load data
# =============================================================================
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)



# =============================================================================
# Select specific rows and columns from orginal data to create a simpler data
# set. This simplifies presentation of perceptron algorithm. 
# =============================================================================

# Select setosa and versicolor. We now that those are linearly seperable
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Extract sepal length and petal length. We need only these two varibles to 
# separate setosa from virginica. Working with only two variables makes it 
# easy for us to plot the data in scatter plots. 
X = df.iloc[0:100, [0, 2]].values


# Plot data from the selected features. 
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_06.png', dpi=300)
plt.show()



# =============================================================================
# Training the adaline model and plot cost functions
# =============================================================================

# Initiate the adaline models (class) with specific parameters for eta and 
# number of iterations.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

ada1 = AdalineGD(n_iter=20, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
#
ada2 = AdalineGD(n_iter=30, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.savefig('images/02_11.png', dpi=300)
plt.show()



# =============================================================================
# Plot decision boundary
# =============================================================================

# First define function that plots decision boundary
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


# Now plot decision boundary of trained  model
plt.figure(0, figsize=(8, 8), dpi=120)
plot_decision_regions(X, y, classifier=ada2)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')


# plt.savefig('images/02_08.png', dpi=300)
plt.show()




