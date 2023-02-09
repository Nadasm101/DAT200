# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 14:38:12 2018

@author: olive
"""


import numpy as np


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        self.weights_ = [self.w_.copy()] # OT: collect inital weights in a list
        print('\ninitial weights:', self.w_.copy()) # OT: print intitial weights
        numUpdates = 0 # OT: initiate counter for number of updates

        for ind in range(self.n_iter): # OT: changed from '_' to 'ind'
            errors = 0
            zipInd = 0 # OT: index for identification of instance loop in zip
            
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                
                if update != 0.0: # OT: When updating happens do the following ...
                    numUpdates += 1
                    self.weights_.append(self.w_.copy()) # OT: append copy of weight in list
                    print('\nUpdate nr. ', numUpdates) #OT: Increase counter at every update
                    print('updated weights:', self.w_.copy()) # OT: print updated weights
                    print('-- epoch', ind + 1, '-- sample', zipInd) # OT: print index for iteration and index for zip
                    print('xi:', xi) # OT: print xi
                
                zipInd = zipInd + 1 # OT: increase zipInd at each row 
            
            self.errors_.append(errors)
        
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)



