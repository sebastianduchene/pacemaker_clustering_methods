# Runing Rmagic from within python scripts

from IPython import get_ipython
import dendropy


ipython = get_ipython()

ipython.magic('%load_ext rmagic')
ipython.run_line_magic('R', 'library(ape)')
ipython.run_line_magic('R', 'tr <- rtree(10)')

wow = ipython.run_line_magic('R', 'write.tree(tr)')

print wow
# The latter save the tree as text in python, which can be read in dendropy and printed to screen:


my_tree = dendropy.Tree.get_from_string(wow[0], schema = 'newick')

my_tree.print_plot()




