# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:17:45 2019

@author: olto
"""

# =============================================================================
# Import modules
# =============================================================================
import numpy.random as npr

# Numpy random module
# https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.random.html

# Numpy random - normal distribution
# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html#numpy.random.normal



# =============================================================================
# Initialise values for weights WITHOUT seed
# =============================================================================
w1 = npr.normal(loc=0.0, scale=0.01, size=5)
w2 = npr.normal(loc=0.0, scale=0.01, size=5)
w3 = npr.normal(loc=0.0, scale=0.01, size=5)

# Conclusion: the weights are different every time npr.normal is called

#
## =============================================================================
## Initialise values for weights WITH seed
## =============================================================================

# Set seed for random state
rs_state_seed = 77

rgen = npr.RandomState(rs_state_seed)
w4 = rgen.normal(loc=0.0, scale=0.01, size=5)

rgen = npr.RandomState(rs_state_seed)
w5 = rgen.normal(loc=0.0, scale=0.01, size=5)

rgen = npr.RandomState(rs_state_seed)
w6 = rgen.normal(loc=0.0, scale=0.01, size=5)

# Conclusion: the weights are identical every time npr.normal is called,
# provided that npr.RandomState is called with same seed every time prior to 
# calilng npr.normal




