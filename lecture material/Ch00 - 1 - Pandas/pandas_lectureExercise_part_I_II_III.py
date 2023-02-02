# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 20:44:51 2019

@author: olto
"""

# Import necessary modules
import pandas as pd
import numpy as np


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



# =============================================================================
# Get subset of df based on conditional
# =============================================================================
df_iris_setosa = df[df['types'] == 'Iris-setosa']
df_iris_versicolor = df[df['types'] == 'Iris-versicolor']
df_iris_virginica = df[df['types'] == 'Iris-virginica']

classCounts = df['types'].value_counts()









# =============================================================================
# View last 10 rows of columns 'sepal length' and 'types'
# =============================================================================

print(df[['sepal length', 'types']].tail(10))



# =============================================================================
# View rows where 'sepal length' > 5 and 'petal width' < 0.2
# =============================================================================
print(df[(df['sepal length'] > 5) & (df['petal width'] < 0.2)])


# Make new df where 'petal width' is exactly 1.8
df_specific = df.where(df['petal width'] == 1.8).dropna()
df_specific_2 = df[df['petal width'] == 1.8]



# =============================================================================
# Get descriptive statistics for the dataframe
# =============================================================================
print(df.describe())

print(df['petal length'].describe())



# =============================================================================
# Remove rows named 'flower 55' and 'flower 77'
# =============================================================================

df = df.drop(['flower 55', 'flower 77'])



# =============================================================================
# # Remove column 'sepal width >= 3'
# =============================================================================

df = df.drop('sepal width >= 3', axis=1)



# =============================================================================
# Find where a specific value exists in a column based on a conditional
# =============================================================================

spec_sepalLength = df['sepal length'].where(df['petal width'] == 1.8)



# =============================================================================
# Get only values dataframe (get rid of column and row names)
# =============================================================================
data = df.values



# =============================================================================
# Apply some function to all numerical cells in df
# =============================================================================

# Get rid of column 'types'. We want to applya function to numerical cells only
df = df.drop('types', axis=1)

# Define what to do with each cell in df. For each cell value x, add 1, then 
# multiply by 3. This does not do any computations. This only defines the 
# function and does not apply it.
computation = lambda x: (x + 1) * 3

# Now apply this function to all cells
df_new = df.applymap(computation)





