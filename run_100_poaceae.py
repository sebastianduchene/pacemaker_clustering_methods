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

## Poacea data 
poac_data = pd.DataFrame(rescale_data(np.array(pd.read_csv('poaceae_matrix.csv'))))
poac_VBGMM = fit_VBGMM(np.array(poac_data))
poac_DPGMM = fit_DPGMM(np.array(poac_data))

samples_poaceae = simulate_data(poac_data, 2, 'spherical', 100)

poaceae_sims = fit_sim_models(samples_poaceae)

poaceae_sims.to_csv('poaceae_simulations.csv')