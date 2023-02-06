# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:42:02 2019

@author: olto
"""



# =============================================================================
# Import modules
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
#from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions



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



# =============================================================================
# Plot decision boundary
# =============================================================================

# Now plot decision boundary of trained perceptron model
plt.figure(2, figsize=(8, 8), dpi=120)
plot_decision_regions(X, y, clf=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')


# plt.savefig('images/02_08.png', dpi=300)
plt.show()


