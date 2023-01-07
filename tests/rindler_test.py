import sys
sys.path.insert(1, '/Users/marcusallen/Documents/GitHub/causal-set-simulations/causal_sets')
from rindler import *
causet = RindlerHorizon(rho=10000)
causet.number_points()
print(causet.n)
causet.coordinates()
above, below, left, right = causet.cuts4(causet.points)
rels_above =get_relations_above(N=causet.n, points=causet.points, above=above)
rels_below =get_relations_below(N=causet.n, points=causet.points, below=below)
count = interval_size(rels_above,rels_below)
SMI = SMI_calc(count, causet.dim, 10)
print(SMI)