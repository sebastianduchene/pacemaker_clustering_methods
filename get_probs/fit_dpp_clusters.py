
print 'loading dependencies: sys, pandas, numpy, sklearn.mixture, matplotlib'

import sys
import pandas as pd
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from matplotlib import pylab
#get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (10.0, 4.0)

print 'dependencies loaded correctly'

def rescale_data(br_matrix):
    mat_scaled = np.empty(shape = br_matrix.shape)
    for r in range(br_matrix.shape[0]):
        mat_scaled[r, :] = np.log10((br_matrix[r, :] / br_matrix[r, :].sum()) )
    return mat_scaled


def fit_DPP(brlen_matrix, gene_names = None):
    
    fit_diag = mixture.DPGMM(n_components = brlen_matrix.shape[0]-2, covariance_type = 'diag', n_iter = 10000).fit(brlen_matrix)
    pred_diag = fit_diag.predict(brlen_matrix)
    prob_diag = fit_diag.predict_proba(brlen_matrix)
    
    fit_sphe = mixture.DPGMM(n_components = brlen_matrix.shape[0]-2, covariance_type = 'spherical', n_iter = 10000).fit(brlen_matrix)
    pred_sphe = fit_sphe.predict(brlen_matrix)
    prob_sphe = fit_sphe.predict_proba(brlen_matrix)
    
    summary_matrix = np.empty(shape = (2, 2))
    summary_matrix[0, :] = [fit_diag.bic(brlen_matrix), len(set(pred_diag))]
    summary_matrix[1, :] = [fit_sphe.bic(brlen_matrix), len(set(pred_sphe))]
    summary_matrix = pd.DataFrame(summary_matrix)
    summary_matrix.columns = ['BIC', 'n_clusters']
    summary_matrix.index = ['diagonal', 'spherical']
    
    if not gene_names is None:
        pred_diag = pd.DataFrame(pred_diag)
        pred_diag.index = gene_names
        pred_diag.columns = ['cluster']
        prob_diag = pd.DataFrame(prob_diag)
        prob_diag.index = gene_names
        
        pred_sphe = pd.DataFrame(pred_sphe)
        pred_sphe.index = gene_names
        pred_sphe.columns = ['cluster']
        prob_sphe = pd.DataFrame(prob_sphe)
        prob_sphe.index = gene_names

        return (summary_matrix, pred_diag, prob_diag, pred_sphe, prob_sphe)    
    else:
        return (summary_matrix, pred_diag, prob_diag, pred_sphe, prob_sphe)

print 'reading file %s' %sys.argv[1]

brlen_file = sys.argv[1]
brlen_frame = pd.read_csv(brlen_file, index_col=0).transpose()

print 'fitting mixture models using Dirichlet process prior'
gene_names = brlen_frame.index
brlen_matrix = rescale_data(np.array(brlen_frame))


ax = pd.DataFrame(brlen_matrix).transpose().plot(legend = False, linewidth = 2)
ax.set_ylabel('log10 branch length')
ax.set_xlabel('branch')
plt.savefig('brlen_clusters.pdf')

mixture_models = fit_DPP(brlen_matrix, gene_names)

mixture_models[0].to_csv('summary_dpp.csv')
mixture_models[1].to_csv('prediction_diagonal.csv')
mixture_models[2].to_csv('probabilities_diagonal.csv')

mixture_models[3].to_csv('prediction_spherical.csv')
mixture_models[4].to_csv('probabilities_spherical.csv')

print 'saved results to: brlen_clusters.pdf, summary_dpp.csv, prediction_diagonal.csv, probabilities_diagonal.csv, prediction_spherical.csv, probabilities_spherical.csv'
