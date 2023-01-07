"""
=========================================
Base Simulation (:mod:`base_simulation`)
=========================================
"""

"""
Includes all import files needed to run
"""
import time
import pickle
import numpy as np
from math import gamma
from matplotlib import rc, rcParams
from scipy.optimize import curve_fit, fsolve
import matplotlib.pyplot as plt
from numba import njit, prange, jit
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg.blas import sgemm

#rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
#rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{amsmath, nicefrac, graphicx}')
rcParams.update({'font.size': 24})

vivid_colors = ['blue', 'darkorange', 'green', 'firebrick', 'indigo',
           'mediumvioletred', 'darkturquoise', 'gold', 'coral', 'darkkhaki']
pale_colors = ['cornflowerblue', 'orange', 'limegreen', 'indianred',
           'mediumpurple', 'hotpink', 'paleturquoise', 'gold', 'lightcoral', 'khaki']

"""
Basic methods not related to simulating but to getting results
"""

def propto(x,a):
    return np.asarray(x)*a
def linear(x,a,b):
    return np.asarray(x)*a+b
def quadratic(x,a,b,c):
    return (np.asarray(x)**2)*a+np.asarray(x)*b+c
def power_law(x,a,b):
    return (np.asarray(x)**a)*b

def StoreData(data_list: list, name_of_pickle: str):
    """ Stores list of data. Overwrites any previous data in the pickle file. """
    # Delete previous data
    pickle_file = open(name_of_pickle, 'w+')
    pickle_file.truncate(0)
    pickle_file.close()
    # Write new data
    pickle_file = open(name_of_pickle, 'ab')  # Mode: append + binary
    pickle.dump(data_list, pickle_file)
    pickle_file.close()
    
def LoadData(name_of_pickle:str):
    pickle_file = open(name_of_pickle, 'rb')  # Mode: read + binary
    data_base = pickle.load(pickle_file)
    pickle_file.close()
    return data_base

def time_to_run(initial, final=None):
    """Print the time to run."""
    t = time.time() - initial if final==None else final - initial
    hrs, rest = divmod(t, 3600)
    mins, secs = divmod(rest, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hrs),int(mins),int(secs))


def n_sphere(dim,n):
    if dim < 1:
        raise ValueError('Dimension must be at least 1')
    points = np.random.randn(n,dim+1)
    for i in range(len(points)):
        while np.linalg.norm(points[i])>1:
            points[i] = np.random.randn(dim+1)
    for i in prange(len(points)):
        points[i] = points[i]/np.linalg.norm(points[i])
    return points
