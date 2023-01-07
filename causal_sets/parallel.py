"""
=========================================
Parallelized functions (:mod:`parallel`)
=========================================
"""

"""
Includes all functions which are parallelized. Excluded from classes on the basis of:
 1. Some are shared across different classes
 2. Numba is not very happy when it is run within classes
"""

"""
Methods for finding relations for regions
"""

from methods import *
"""
Causality methods
"""
@njit(fastmath=True)
def Causal_future(p1,p2):
    d_t = (p2[0]-p1[0])
    d_x= (p1[1]-p2[1])
    if d_t**2 >= (d_x**2):
        return True
    else:
        return False

@njit(fastmath=True)
def Causal_future_wrapped(p1,p2, circ):
    d_t = (p2[0]-p1[0])
    d_x= (p1[1]-p2[1])
    d_y = (p1[2]-p2[2])
    
    if d_t**2 >= (d_x**2+d_y**2):
        return True
    elif d_t**2 >= (d_y**2+(p1[1]-(p2[1]-circ))**2):
        return True
    else:
        return False

"""
Spliting relation matrix
"""
#@njit(fastmath=True,parallel=True)
def split_array(N, above, below):
    max_size = 240000000//N
    above_split = len(above)//max_size
    above_spare = len(above) - above_split*max_size
    below_split = len(below)//max_size
    below_spare = len(below) - below_split*max_size
    split_above_list = np.insert(np.linspace(above_spare,above_spare+above_split*max_size,above_split+1),0,0).astype(np.int64)
    split_below_list = np.insert(np.linspace(below_spare,below_spare+below_split*max_size,below_split+1),0,0).astype(np.int64)
    return int(above_split), int(below_split), int(above_spare), int(below_spare), int(max_size), split_above_list,split_below_list


"""
Getting relations
"""

@njit(fastmath=True,parallel=True)
def get_relations_above(N, points, above):
    rels_above = np.zeros((N,len(above)),dtype=np.float32)
    for i in prange(len(above)):
        for k in prange(above[i]+1):
            if Causal_future(points[above[i]],points[k]):
                rels_above[k,i] = 1
    return rels_above

@njit(fastmath=True,parallel=True)
def get_relations_below(N, points, below):
    rels_below = np.zeros((len(below),N),dtype=np.float32)
    for j in prange(len(below)):
        for k in prange(len(points[below[j]:])):
            i = k+below[j]
            if Causal_future(points[i],points[below[j]]):
                rels_below[j,i] = 1
    return rels_below

#Wrapped
@njit(fastmath=True,parallel=True)
def get_relations_above_wrapped(N, points, above, circ):
    rels_above = np.zeros((N,len(above)),dtype=np.ushort)
    for i in prange(len(above)):
        for k in prange(above[i]+1):
            if Causal_future_wrapped(points[above[i]],points[k], circ):
                rels_above[k,i] = 1
    return rels_above

@njit(fastmath=True,parallel=True)
def get_relations_below_wrapped(N, points, below, circ):
    rels_below = np.zeros((len(below),N),dtype=np.ushort)#uint
    for j in prange(len(below)):
        for k in prange(len(points[below[j]:])):
            i = k+below[j]
            if Causal_future_wrapped(points[i],points[below[j]],circ):
                rels_below[j,i] = 1  
    return rels_below

#Split
def get_rels_above_split(split_above_list, above_split,N, points, above, circ):
    for s in range(above_split+1):
        StoreData(np.packbits(get_relations_above_wrapped(N, points, above[split_above_list[s]:split_above_list[s+1]], circ), axis=None),"above_rels"+str(s)) #.astype(np.ubyte)

def get_rels_below_split(split_below_list, below_split,N, points, below, circ):
    for s in range(below_split+1):
        StoreData(np.packbits(get_relations_below_wrapped(N, points, below[split_below_list[s]:split_below_list[s+1]], circ), axis=None),"below_rels"+str(s))


"""
Methods for determining the interval sizes - non-gpu
"""

@njit(fastmath=True)
def interval_size(rels_above,rels_below):
    return np.bincount((rels_below@rels_above).astype(np.int64).flatten())[2:]


"""
SMI functions
"""
@njit(fastmath=True)
def f2(n,eps):
    beta_d = 4
    eps_d = eps**2
    f = ((1-eps)**n)*(1-((2*eps*n)/(1-eps))+(((eps**2)*n*(n-1))/(2*((1-eps)**2))))
    return beta_d*eps_d*f

@njit(fastmath=True)
def f3(n,eps):
    beta_d = ((np.pi/(3*(2**0.5)))**(2/3))/gamma(5/3)
    eps_d = eps**(5/3)
    f = ((1-eps)**n)*(1-((27*eps*n)/(8*(1-eps)))+((9*(eps**2)*n*(n-1))/(8*((1-eps)**2))))
    return beta_d*eps_d*f

@njit(fastmath=True)
def f4(n,eps):
    beta_d = 4/(6**0.5)
    eps_d = eps**(3/2)
    f = ((1-eps)**n)*(1-((9*eps*n)/(1-eps))+((8*(eps**2)*n*(n-1))/(((1-eps)**2)))
                    -((4*(eps**3)*n*(n-1)*(n-2))/(3*((1-eps)**3))))
    return beta_d*eps_d*f

"""
Methods for calculating the SMI
"""

@njit(fastmath=True)
def SMI_calc(count, dim, smearing_scale):
    
    """ Calculates the SMI of the causal set across the horizon. Based on 'The Action of a Causal Set' by Benincasa.
        · smearing_scale: smearing scale as a multiple of the discreteness scale."""
    eps = (1/smearing_scale)**dim
    
    if dim == 2:
        SMI = sum([num*f2(n,eps) for n, num in enumerate(count)])
    if dim == 3:
        SMI = sum([num*f3(n,eps) for n, num in enumerate(count)])
    if dim == 4:
        SMI = sum([num*f4(n,eps) for n, num in enumerate(count)])
    return SMI

@njit(fastmath=True,parallel=True)
def l_k_range_measurements(SMI,l_ks, count,dim):
    for k in prange(len(l_ks)):
        SMI[k] = SMI_calc(count=count,dim=dim, smearing_scale=l_ks[k])