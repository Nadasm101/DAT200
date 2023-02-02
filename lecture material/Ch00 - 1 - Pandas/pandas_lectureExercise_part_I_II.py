# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 20:44:51 2019

@author: olto
"""

# Import necessary modules
import pandas as pd
import numpy as np


# =============================================================================
# Load data
# =============================================================================
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data', header=None)



# =============================================================================
# Rename multiple pandas Dataframe Column Names
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



# =============================================================================
# Find Unique Values In Pandas Dataframes
# =============================================================================
# https://chrisalbon.com/python/data_wrangling/pandas_find_unique_values/
unique_classes_alt1 = list(set(df.types))
unique_classes_alt2 = list(df['types'].unique())



# =============================================================================
# Grouping rows in pandas
# =============================================================================
# https://chrisalbon.com/python/data_wrangling/pandas_group_rows_by/
df_byClass = df.groupby(df['types'])
classMeans = df_byClass.mean()

# Alternative in one line
classMeans2 = df.groupby(df['types']).mean()



# =============================================================================
# Create a Column Based on a Conditional in pandas
# =============================================================================
# https://chrisalbon.com/python/data_wrangling/pandas_create_column_using_conditional/
df['sepal width >= 3'] = np.where(df['sepal width'] >= 3.0, True, False)

# Count number of those with sepal width 3.0 cm or over
df['sepal width >= 3'].sum()




