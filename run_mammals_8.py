import os, sys, re
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import mixture
import scipy.stats as sp
from scipy.spatial import distance
import scipy.stats as sc
from GMM_trees import *
import itertools

execfile('GMM_trees.py')

## Load mammal data simulations
mm_dat = pd.DataFrame(rescale_data(np.array(pd.read_csv('mammal_matrix.csv'))))

# Simulate 100 data sets
samples_temp_8 = simulate_data(mm_dat, 8, 'spherical', 100)
for i in range(len(samples_temp_8)):
    pd.DataFrame(samples_temp_8[i]).to_csv('sim_mammals_8/sim_'+str(i)+'.csv')

print 'fitting mammals 8'
mm_sims_8 = fit_sim_models(samples_temp_8)
mm_sims_8.to_csv('mm_sims_8.csv')


