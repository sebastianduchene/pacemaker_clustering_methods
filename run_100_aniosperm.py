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

## Angiosperm simulations
angio_dat = pd.DataFrame(np.array(pd.read_csv('angiosperm_matrix.csv')))
angio_vbgmm1 = fit_VBGMM(np.array(angio_dat))
angio_dppgmm1 = fit_DPGMM(np.array(angio_dat))

# Simulate 100 data sets
samples_temp = simulate_data(angio_dat, 1, 'spherical', 100)

angio_sims = fit_sim_models(samples_temp)
angio_sims.to_csv('angio_simulations.csv')

angio_sims = fit_sim_models(samples_temp)
angio_sims.to_csv('angio_simulations.csv')

