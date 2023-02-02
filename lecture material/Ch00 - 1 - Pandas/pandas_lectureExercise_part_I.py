# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 20:44:51 2019

@author: olto
"""

# =============================================================================
# Import necessary modules
# =============================================================================
import pandas as pd



# =============================================================================
# # Load data
# =============================================================================
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data', header=None)


# =============================================================================
# # Rename multiple pandas Dataframe Column Names
# =============================================================================
# Define column names
colnames = ['sepal length', 
            'sepal width', 
            'petal length', 
            'petal width',
            'types']

# Set column names 
df.columns = colnames

# Quick look what colum names and row names look like at this point
print(df.columns)
print(df.index)

# Generate row names
flowerInd = ['flower {0}'.format(str(x + 1)) for x in range(150)]

# Set row names
df.index = flowerInd




