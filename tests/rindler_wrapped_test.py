import sys
sys.path.insert(1, '/home/marcus/causal-set-simulations/causal_sets')
from rindler import *
from parallel import *
from GPU import *

code_start =time.time()

causet = RindlerHorizon(rho=3000, dim = 3, circ = 8) #15000
causet.number_points()
print(causet.n)
causet.coordinates()
above, below, left, right = causet.cuts4(causet.points)

get_rels =time.time()
rels_above =get_relations_above_wrapped(N=causet.n, points=causet.points, above=above, circ = causet.circ)
rels_below =get_relations_below_wrapped(N=causet.n, points=causet.points, below=below, circ = causet.circ)
print('Time to get relations',time_to_run(get_rels))

start5 =time.time()
intervals = (interval_size_gpu(rels_above,rels_below))
print('Time to run multiplication',time_to_run(start5))

max_lk = 20
numb_lk = 40
l_ks = np.linspace(1.00025,max_lk,numb_lk)

#SMI = np.zeros((numb_lk))
#l_k_range_measurements(SMI=SMI, l_ks=l_ks,count=intervals,dim=causet.dim)
#print(SMI)

print('Time to run whole script',time_to_run(code_start))

