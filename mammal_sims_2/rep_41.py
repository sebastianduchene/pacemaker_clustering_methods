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
import multiprocessing
from multiprocessing import Pool

execfile('GMM_trees.py')
execfile('functions.py')

# Simulate 100 data sets
mm_dat = pd.DataFrame(rescale_data(np.array(pd.read_csv('mammal_matrix.csv'))))

samples_temp = simulate_data(mm_dat, 2, 'spherical', 1)

f = fit_data(samples_temp[0])
open('out_s1_41.txt', 'w').write(f[0])
