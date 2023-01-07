#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:41:31 2022

@author: marcus
"""

import sys
sys.path.insert(1, '/home/marcus/causal-set-simulations/causal_sets')
from rindler import *
from parallel import *
from GPU import *

code_start =time.time()

for t in range(0,10):
    
    causet = RindlerHorizon(rho=37500, dim = 3, circ = 6) 
    causet.number_points()
    print(causet.n)
    causet.coordinates()
    above, below, left, right = causet.cuts4(causet.points)

    get_rels=time.time()
    above_split, below_split, above_spare, below_spare, max_size, split_above_list,split_below_list = split_array(N = causet.n, above=above, below=below)
    get_rels_above_split(split_above_list, above_split,N=causet.n, points=causet.points, above=above, circ = causet.circ)
    get_rels_below_split(split_below_list, below_split,N=causet.n, points=causet.points, below=below, circ = causet.circ)
    print('Time to get relations new',time_to_run(get_rels))

    start5 =time.time()
    intervals = np.zeros(1, dtype = np.int64)
    for j in range(int(below_split+1)):
        below_use = LoadData("below_rels"+str(j))
        if j == 0:
            below_use = np.unpackbits(below_use, count=(below_spare*causet.n)).reshape((below_spare,causet.n)).astype(np.short)
        else:
            below_use = np.unpackbits(below_use, count=(max_size*causet.n)).reshape((max_size,causet.n)).astype(np.short)
        for i in range(int(above_split+1)):
            above_use = LoadData("above_rels"+str(i))
            if i == 0:
                above_use = np.unpackbits(above_use, count=(above_spare*causet.n)).reshape((causet.n,above_spare)).astype(np.short)
            else:
                above_use = np.unpackbits(above_use, count=(max_size*causet.n)).reshape((causet.n,max_size)).astype(np.short)
            interval_sub = interval_size_gpu(above_use,below_use)
            if len(intervals)> len(interval_sub):
                interval_sub.resize(len(intervals))
                intervals = np.add(intervals,interval_sub)
            else:
                intervals.resize(len(interval_sub))
                intervals = np.add(intervals,interval_sub)
        print(j/int(below_split+1))
    print('Time to run multiplication new',time_to_run(start5))
    
    StoreData(intervals,"rindlerwrapped3D_"+"circ"+str(causet.circ)+"_rho"+str(causet.rho)+"_"+str(t))
print("total time ", time_to_run(code_start))
