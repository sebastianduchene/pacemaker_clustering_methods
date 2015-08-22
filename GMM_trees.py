import os, sys, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn import mixture
import scipy.stats as sp
from GMM_trees import *
import itertools


def rescale_data(br_matrix):
    mat_scaled = np.empty(shape = br_matrix.shape)
    for r in range(br_matrix.shape[0]):
        mat_scaled[r, :] = np.log10((br_matrix[r, :] / br_matrix[r, :].sum()) )
    return mat_scaled


def fit_GMM(br_matrix, svd_transform = False, standard_matrix = False):
    
    if standard_matrix:
        st_mat = br_matrix
        for r in range(br_matrix.shape[0]):
            st_mat[r, :] = np.log10(br_matrix[r, :] / br_matrix[r, :].sum())
        data_fit = st_mat

    if svd_transform:
        svd_data = np.linalg.svd(br_matrix)
        n_dim = np.where( (svd_data[1] / svd_data[1].sum()).cumsum() > 0.95 )[0][0] + 1
        n_dim = 3
        print (svd_data[1] / svd_data[1].sum()).cumsum()
        print n_dim
        data_fit = svd_data[0]
    else:
        n_dim = br_matrix.shape[1]
        data_fit = br_matrix

    cov_matrix = ['diag', 'spherical']
    models_fit = []
    prediction_results = {}
    fit_results = np.empty(shape = (4, 4))

    # Find optimal components with bic:
    diag_models = []
    spher_models = []
    for n_c in range(1, data_fit.shape[0] - 3):
        if n_c % 10 == 0:
            print 'fitting %s clusters' %n_c
        diag_models.append(mixture.GMM(n_components = n_c, covariance_type = 'diag', n_iter = 1000, n_init = 5).fit(data_fit))
        spher_models.append(mixture.GMM(n_components = n_c, covariance_type = 'spherical', n_iter = 1000, n_init = 5).fit(data_fit))

    diag_bics = [m.bic(data_fit) for m in diag_models]
    spher_bics = [m.bic(data_fit) for m in spher_models]

    diag_best = diag_models[np.where(diag_bics == np.min(diag_bics))[0]]
    spher_best = spher_models[np.where(spher_bics == np.min(spher_bics))[0]]
    
    results_matrix = pd.DataFrame(np.empty(shape = (2, 4)))
    results_matrix.ix[:, 0] = ['diagonal', 'spherical']
    results_matrix.ix[:, 1] = [diag_bics[np.where(diag_bics == np.min(diag_bics))[0]], spher_bics[np.where(spher_bics == np.min(spher_bics))[0]]]
    results_matrix.ix[:, 2] = [len(set(diag_best.predict(data_fit))), len(set(spher_best.predict(data_fit))) ]
    results_matrix.ix[:, 3] = [diag_best.n_components, spher_best.n_components]
    results_matrix.columns = ['covariance_type', 'BIC', 'clusters_used', 'n_components']

    return [diag_best, spher_best, results_matrix]




def fit_VBGMM(br_matrix, svd_transform = False, standard_matrix = False):

    if standard_matrix:
        st_mat = br_matrix
        for r in range(br_matrix.shape[0]):
            st_mat[r, :] = np.log10(br_matrix[r, :] / br_matrix[r, :].sum())
        data_fit = st_mat

    if svd_transform:
        svd_data = np.linalg.svd(br_matrix)
        n_dim = np.where( (svd_data[1] / svd_data[1].sum()).cumsum() > 0.95 )[0][0] + 1
        n_dim = 3
        print (svd_data[1] / svd_data[1].sum()).cumsum()
        print n_dim
        data_fit = svd_data[0]
    else:
        n_dim = br_matrix.shape[1]
        data_fit = br_matrix

    cov_matrix = ['diag', 'spherical']
    models_fit = []
    prediction_results = {}
    fit_results = np.empty(shape = (4, 4))

    diag_models = []
    spher_models = []
    for n_c in range(1, data_fit.shape[0] - 3):
        if n_c % 10 == 0:
            print 'fitting %s clusters' %n_c
        diag_models.append(mixture.VBGMM(n_components = n_c, covariance_type = 'diag', n_iter = 1000).fit(data_fit))
        spher_models.append(mixture.VBGMM(n_components = n_c, covariance_type = 'spherical', n_iter = 1000).fit(data_fit))

    diag_bics = [m.bic(data_fit) for m in diag_models]
    spher_bics = [m.bic(data_fit) for m in spher_models]

    diag_best = diag_models[np.where(diag_bics == np.min(diag_bics))[0]]
    spher_best = spher_models[np.where(spher_bics == np.min(spher_bics))[0]]

    results_matrix = pd.DataFrame(np.empty(shape = (2, 4)))
    results_matrix.ix[:, 0] = ['diagonal', 'spherical']
    results_matrix.ix[:, 1] = [diag_bics[np.where(diag_bics == np.min(diag_bics))[0]], spher_bics[np.where(spher_bics == np.min(spher_bics))[0]]]
    results_matrix.ix[:, 2] = [len(set(diag_best.predict(data_fit))), len(set(spher_best.predict(data_fit))) ]
    results_matrix.ix[:, 3] = [diag_best.n_components, spher_best.n_components]
    results_matrix.columns = ['covariance_type', 'BIC', 'clusters_used', 'n_components']

    return [diag_best, spher_best, results_matrix]



def fit_DPGMM(br_matrix, svd_transform = False, standard_matrix = False):

    if standard_matrix:
        st_mat = br_matrix
        for r in range(br_matrix.shape[0]):
            st_mat[r, :] = np.log10(br_matrix[r, :] / br_matrix[r, :].sum())
        data_fit = st_mat

    if svd_transform:
        svd_data = np.linalg.svd(br_matrix)
        n_dim = np.where( (svd_data[1] / svd_data[1].sum()).cumsum() > 0.95 )[0][0] + 1
        n_dim = 3
        print (svd_data[1] / svd_data[1].sum()).cumsum()
        print n_dim
        data_fit = svd_data[0]
    else:
        n_dim = br_matrix.shape[1]
        data_fit = br_matrix

    cov_matrix = ['diag', 'spherical']
    models_fit = []
    prediction_results = {}
    fit_results = np.empty(shape = (4, 4))

    diag_best = mixture.DPGMM(n_components = data_fit.shape[0] -1, covariance_type = 'diag', n_iter = 1000).fit(data_fit)
    spher_best = mixture.DPGMM(n_components = data_fit.shape[0] - 1, covariance_type = 'spherical', n_iter = 1000).fit(data_fit)
    
    results_matrix = pd.DataFrame(np.empty(shape = (2, 4)))
    results_matrix.ix[:, 0] = ['diagonal', 'spherical']
    results_matrix.ix[:, 1] = [diag_best.bic(data_fit), spher_best.bic(data_fit)]
    results_matrix.ix[:, 2] = [len(set(diag_best.predict(data_fit))), len(set(spher_best.predict(data_fit))) ]
    results_matrix.ix[:, 3] = [diag_best.n_components, spher_best.n_components]
    results_matrix.columns = ['covariance_type', 'BIC', 'clusters_used', 'n_components']

    return [diag_best, spher_best, results_matrix]
    
    
def simulate_data(fit_data, n_comp, cov_type, n_sims):
    
    from sklearn import mixture
    
    model_fit = mixture.GMM(n_components = n_comp, covariance_type = cov_type).fit(fit_data)
    
    sim_datasets = []
    for i in range(n_sims):
        sim_datasets.append(model_fit.sample(fit_data.shape[0]))
        
    return sim_datasets

def unlist(nested_list):
    if any([type(i) == list for i in nested_list]):
        out_list = list()
        for e in nested_list:
            if type(e) == list:
                for s in e:
                    out_list.append(s)
            else:
                out_list.append(e)
        return out_list
    else: 
        return nested_list
        

def fit_sim_models(simulated_data):
    results_mat = pd.DataFrame(np.empty(shape = (len(simulated_data), 2)))

    for i in range(len(simulated_data)):
        fit_temp_vbgmm = fit_VBGMM(simulated_data[i])
        fit_temp_dpp = fit_DPGMM(simulated_data[i])

        run_models = unlist([fit_temp_vbgmm[:2], fit_temp_dpp[:2]])
        models_order = np.hstack([fit_temp_vbgmm[2]['BIC'], fit_temp_dpp[2]['BIC']]).argsort()
        best_model = run_models[models_order[0]]

        k_best_model = len(set(best_model.predict(simulated_data[i])))
        name_best_model = ['vbgmm_diag', 'vbgmm_shperical', 'dpp_diag', 'dpp_spherical'][models_order[0]]
    
        results_mat.ix[i, :] = [name_best_model, k_best_model]

    return results_mat
    
def get_mode(data_list):
    from scipy import stats

    freqs = stats.itemfreq(data_list)
    freqs_sorted = freqs[(-freqs[:, 1]).argsort()]
    best_model = freqs_sorted[0, 0]
    prob_model = freqs_sorted[0, 1] / float(freqs_sorted[:, 1].sum())

    return [best_model, round(prob_model, 2)]
