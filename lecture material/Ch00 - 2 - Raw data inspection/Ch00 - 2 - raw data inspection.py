# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:22:23 2022

@author: olive
"""


# =============================================================================
# Import modules and data
# =============================================================================
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import math as m



# =============================================================================
# Load data and define plot settings
# =============================================================================
data = load_breast_cancer()

sns.set(style="whitegrid")



# =============================================================================
# Construct pandas dataframe from objects
# =============================================================================

# Select number of variables to be plotted
num_vars = 4

# Construct data frame wich selected number of components
data_df = pd.DataFrame(data['data'][:, :num_vars])
data_df.columns = data['feature_names'][:num_vars]
data_df['class'] = np.where(data['target'] == 0, 'malignant', 'beningn')



# =============================================================================
# Descriptive statistics
# =============================================================================

decsr_stats = data_df.describe()



# =============================================================================
# Histograms
# =============================================================================
data_df.hist()
plt.show()



# =============================================================================
# Density plots
# =============================================================================
data_df.plot(kind='density', 
                 subplots=True, 
                 layout=(int(m.sqrt(num_vars)), int(m.sqrt(num_vars))), 
                 sharex=False)
plt.show()



# =============================================================================
# Box and Whisker Plots
# =============================================================================
data_df.plot(kind='box', 
                 subplots=True, 
                 layout=(int(m.sqrt(num_vars)), int(m.sqrt(num_vars))), 
                 sharex=False, 
                 sharey=False)
plt.show()



# =============================================================================
# Violin plot
# =============================================================================

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 6))

# Draw violin plot
sns.violinplot(data=data_df, palette="Set3", bw=.2, cut=1, linewidth=1)

#ax.set(ylim=(-.7, 1.05))
sns.despine(left=True, bottom=True)
plt.show()



# =============================================================================
# Plot correlation matrix
# =============================================================================
correlations = data_df.corr()

fig = plt.figure()
ax = fig.add_subplot(111)
fig, ax = plt.subplots(figsize=(11, 6))
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, num_vars, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(list(data_df.columns[:-1]))
ax.set_yticklabels(list(data_df.columns[:-1]))
plt.show()



# =============================================================================
# Scatter plot matrix (pandas) or pairplot (seaborn)
# =============================================================================
# With pandas
pd.plotting.scatter_matrix(data_df)
plt.show()


# Pairplot with seaborn
sns.pairplot(data_df, hue='class')
plt.show()

