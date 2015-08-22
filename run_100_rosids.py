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

rosids_data = pd.DataFrame(rescale_data(np.array(pd.read_csv('rosids_matrix.csv'))))
rosids_VGMM = fit_VBGMM(np.array(rosids_data))
rosids_DPP = fit_DPGMM(np.array(rosids_data))

rosids_samples = simulate_data(rosids_data, 2, 'spherical', 100)

rosids_simulations = fit_sim_models(rosids_samples)

rosids_simulations.to_csv('rosids_simulations.csv')