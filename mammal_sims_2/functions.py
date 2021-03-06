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

def fit_data(s_data):
   fit_temp_vbgmm = fit_VBGMM(s_data)
   fit_temp_dpp = fit_DPGMM(s_data)
   run_models = unlist([fit_temp_vbgmm[:2], fit_temp_dpp[:2]])
   models_order = np.hstack([fit_temp_vbgmm[2]['BIC'], fit_temp_dpp[2]['BIC']]).argsort()
   best_model = run_models[models_order[0]]

   k_best_model = len(set(best_model.predict(s_data)))
   name_best_model = ['vbgmm_diag', 'vbgmm_shperical', 'dpp_diag', 'dpp_spherical'][models_order[0]]
   return [name_best_model, k_best_model]    

def fit_data(s_data):
   fit_temp_vbgmm = fit_VBGMM(s_data)
   fit_temp_dpp = fit_DPGMM(s_data)
   run_models = unlist([fit_temp_vbgmm[:2], fit_temp_dpp[:2]])
   models_order = np.hstack([fit_temp_vbgmm[2]['BIC'], fit_temp_dpp[2]['BIC']]).argsort()
   best_model = run_models[models_order[0]]

   k_best_model = len(set(best_model.predict(s_data)))
   name_best_model = ['vbgmm_diag', 'vbgmm_shperical', 'dpp_diag', 'dpp_spherical'][models_order[0]]
   return [name_best_model, k_best_model]
    

