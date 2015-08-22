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

## Asteraceae data
aster_data = pd.DataFrame(rescale_data(np.array(pd.read_csv('asteraceae_matrix.csv'))))
aster_VBGMM = fit_VBGMM(np.array(aster_data))
aster_DPGMM = fit_DPGMM(np.array(aster_data))

asteraceae_samples = simulate_data(aster_data, 3, 'spherical', 100)

asteraceae_simulations = fit_sim_models(asteraceae_samples)

asteraceae_simulations.to_csv('asteraceae_simulations.csv')
