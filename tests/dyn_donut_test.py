import sys
sys.path.insert(1, '/Users/marcusallen/Documents/GitHub/causal-set-simulations/causal_sets')
from dynamical_donut import *

start1 =time.time()
causet = DynamicalDonutHorizon(rho=460, dim = 3, spr_rad = 2, hor_rad = 1)
causet.number_points()
print(causet.n)
causet.coordinates()
above, below = causet.cuts2(causet.points,causet.hor_rad,causet.spr_rad)
print(causet.proportion)
rels_above =get_relations_above(N=causet.n, points=causet.points, above=above)
rels_below =get_relations_below(N=causet.n, points=causet.points, below=below)
count = interval_size(rels_above,rels_below)
print('Time to run',time_to_run(start1))
amount = 300#len(count)

fig, ax = plt.subplots(1,1,figsize=(12,8))
plt.bar(np.linspace(0,amount,amount), count[:amount], color='royalblue', label = "Original")
plt.ylabel("Count", labelpad=-12.5)
plt.xlabel("Interval size")
plt.legend()
plt.title("2:1 ratio of r:R")