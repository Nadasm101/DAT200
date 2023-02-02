# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 08:46:23 2020

@author: olto
"""


# =============================================================================
# Import modules
# =============================================================================

from sklearn.datasets import load_iris
import pandas as pd



# =============================================================================
# Load data
# =============================================================================

data = load_iris()



# =============================================================================
# Create dataframe from data
# =============================================================================

iris_df = pd.DataFrame(data['data'])
iris_df.columns = data['feature_names']
iris_df.index = [i for i in range(1, 151)]
iris_df['target'] = data['target']



