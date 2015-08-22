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

## Eudicot data
eudicot_dat = pd.DataFrame(rescale_data(np.array(pd.read_csv('eudicot_matrix.csv'))))
eudicot_VBGMM = fit_VBGMM(np.array(eudicot_dat))
eudicot_DPP = fit_DPGMM(np.array(eudicot_dat))

eudicot_samples = simulate_data(eudicot_dat, 2, 'spherical', 100)

eudicot_sims = fit_sim_models(eudicot_samples)
eudicot_sims.to_csv('eudicot_simulations.csv')
