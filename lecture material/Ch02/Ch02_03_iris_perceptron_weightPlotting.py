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
from perceptron_ext import Perceptron
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
plt.figure(0, figsize=(8, 8), dpi=120)
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
# Training the perceptron model
# =============================================================================

# Initiate the perceptron model (class) with specific parameters for eta and 
# number of iterations
ppn = Perceptron(eta=0.1, n_iter=10)


# Train the perceptron model
ppn.fit(X, y)


# Extract classification errors and plot how the error changes across epochs. 
plt.figure(1, figsize=(8, 8), dpi=120)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()


# Print number of updates in each iteration
print(ppn.errors_)

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
#        print('***', idx, cl)
#        print(X[y == cl, 0])
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


# Now plot decision boundary of trained perceptron model
plt.figure(2, figsize=(8, 8), dpi=120)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')


# plt.savefig('images/02_08.png', dpi=300)
plt.show()



# =============================================================================
# Plotting adaption of weights - works only when using perceptron class from
# module perceptron_OT.py
# =============================================================================

# Get the weights after each update. Column 0 holds weights for w0, column 1
# holds values for w1, etc. 
weight_updates = np.vstack(ppn.weights_)


# Generate names for weights, that is w0, w1, w2, etc. Those are needed for 
# the legend in the plot below.
weightsNames = []
for numW in range(np.shape(weight_updates)[1]):
    weightsNames.append('w{0}'.format(numW))
    

# Plot data of each column/weight. Plot shows how weights changed after each
# update.
plt.figure(3, figsize=(8, 8), dpi=120)
for colInd in range(np.shape(weight_updates)[1]):
    plt.plot(weight_updates[:, colInd], label='{0}'.format(weightsNames[colInd]))
leg = plt.legend(loc='best')
plt.xlabel('Number of updates')
plt.ylabel('weight coefficients')
plt.show()


