import os, sys, re
import multiprocessing
from multiprocessing import Pool
execfile('GMM_trees.py')


folder = sys.argv[1]

files = [i for i in os.listdir(folder) if '.csv' in i]

# Load files
sim_list = []
for f in files:
    sim_list.append(pd.read_csv(folder+'/'+f, index_col = 0))



p = Pool(3)

#only for ten so far!!
s = p.map(fit_sim_data, sim_list[0:10])

pd.DataFrame(s).to_csv(folder+'_mixture_models.csv')

print s




