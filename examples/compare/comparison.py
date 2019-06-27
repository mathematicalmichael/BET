import bet.postProcess.compareP as compP
from helpers import *

"""
The ``helpers.py`` file contains functions that define
sample sets of an arbitrary dimension with probabilities
uniformly distributed in a hypercube of sidelength ``delta``.
The hypercube can be in three locations:
- corner at [0, 0, ..., 0]  in ``unit_center_set``
- origin in ``unit_bottom_set``
- corner in [1,1, ..., 1] in `` unit_top_set``

and the number of samples will determine the fidelity of the
approximation since we are using voronoi-cell approximations.
"""
num_samples_left = 50
num_samples_right = 50
delta = 0.5 # width of measure's support per dimension
dim = 2
# define two sets that will be compared
L = unit_center_set(dim, num_samples_left, delta)
R = unit_center_set(dim, num_samples_right, delta)

# choose a reference sigma-algebra to compare both solutions
# against (using nearest-neighbor query).
num_emulation_samples = 2000 
# the compP.compare method instantiates the compP.comparison class.
mm = compP.compare(L, R, num_emulation_samples) # initialize metric

from scipy.stats import entropy as kl_div

mm.set_left(unit_center_set(2, 1000, delta/2))
mm.set_right(unit_center_set(2, 1000, delta))
print([mm.value(kl_div),
       mm.value('tv'),
       mm.value('totvar'),
       mm.value('mink', w=0.5, p=1),
       mm.value('norm'),
       mm.value('sqhell'),
       mm.value('hell'),
       mm.value('hellinger')])