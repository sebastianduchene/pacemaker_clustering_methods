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
#mm_VGMM = fit_VBGMM(np.array(mm_dat))
#mm_DPGMM = fit_DPGMM(np.array(mm_dat))

# Simulate 100 data sets
samples_temp_2 = simulate_data(mm_dat, 2, 'spherical', 100)
for i in range(len(samples_temp_2)):
    pd.DataFrame(samples_temp_2[i]).to_csv('sim_mammals_2/sim_'+str(i)+'.csv')

print 'fitting simulations mammals 2'
mm_sims_2 = fit_sim_models(samples_temp_2)
mm_sims_2.to_csv('mm_sims_2.csv')


