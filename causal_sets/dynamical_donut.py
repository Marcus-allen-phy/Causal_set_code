"""
=============================================================
Dynamical Donut Horizon Simulation (:mod:`dynamical_donut`)
=============================================================
"""

import random
from base_simulation import *

class DynamicalDonutHorizon(BaseSimulation):
    """Dynamical Donut Horizon Simulation class.
    
    Dynamical shell model but with boundary defined by spacelike geodesics above and below the horizon.

    TODO describe Dynamical Donut Horizon
    """
    def __init__(self, dim: int = 2, rho: int = 500, hor_rad: int = 1, spr_rad: int = 1, poisson: bool = True):
        self.dim = dim
        if self.dim<2:
            raise ValueError('Dimension must be at least 2')
        self.rho = rho
        self.hor_rad = hor_rad
        self.circ = (self.hor_rad)*2*np.pi
        self.spr_rad = spr_rad
        self.proportion = (self.spr_rad)/(self.spr_rad+self.hor_rad)
        if self.proportion>2/3:
            raise ValueError('Mus be at most a ratio of 2:1 for spr_rad:hor_rad')
        self.poisson = poisson
        if self.proportion <= 0.5:
            if self.dim  == 2:
                self.volume = 2*(0.5*self.spr_rad*(2*(self.spr_rad+self.hor_rad)+2*self.hor_rad))
            if self.dim == 3:
                tot_cone = (np.pi/3)*(self.spr_rad+self.hor_rad)**3
                inner_tot_cone = (np.pi/3)*(self.hor_rad)**3
                #inner_mini_cone = (np.pi/3)*(self.hor_rad-self.spr_rad)**3
                #self.volume = 2*(tot_cone+inner_mini_cone-2*inner_tot_cone)
                self.volume = 2*(tot_cone-inner_tot_cone)
        if self.proportion >0.5:
            if self.dim == 2:
                self.volume = 2*(0.5*self.spr_rad*(2*(self.spr_rad+self.hor_rad)+2*self.hor_rad))
            if self.dim == 3:
                tot_cone = (np.pi/3)*(self.spr_rad+self.hor_rad)**3
                inner_tot_cone = (np.pi/3)*(self.hor_rad)**3
                #inner_mini_cone = (np.pi/3)*(self.hor_rad-self.spr_rad)**3
                #self.volume = 2*(tot_cone+inner_mini_cone-2*inner_tot_cone)
                self.volume = 2*(tot_cone-inner_tot_cone)

    def coordinates(self):
        self.height = self.hor_rad+self.spr_rad
        maxheight = (self.height)**(self.dim)
        h = (np.random.uniform((((maxheight**(1/self.dim))-(self.height*(self.proportion)))**self.dim),maxheight,self.n)**(1/self.dim))
        t = (maxheight**(1/self.dim)-h)*np.random.choice(np.array([-1, 1]), self.n)
        r = h*(np.random.uniform(0,1,self.n)**(1/(self.dim-1)))
        
        #other = np.random.uniform(-widthy,widthy,(dim-2,n))
        if self.dim == 2:
            x_coords = r*np.random.choice(np.array([-1, 1]), self.n)
            coords = np.vstack((t,x_coords))
        elif self.dim ==3:
            x_coords = n_sphere(self.dim-2, self.n)
            for i in prange(len(x_coords)):
                x_coords[i] = (x_coords[i].T*r[i])
            x_coords = x_coords.T
            coords = np.vstack((t,x_coords[0],x_coords[1]))
        elif self.dim ==4:
            x_coords = n_sphere(self.dim-2, self.n)
            for i in prange(len(x_coords)):
                x_coords[i] = (x_coords[i].T*r[i])
            x_coords = x_coords.T
            coords = np.vstack((t,x_coords[0],x_coords[1],x_coords[2]))
        self.coords = coords[:,np.argsort(coords[0])]
        self.points = self.coords.T

    @staticmethod
    @njit(fastmath=True)
    def cuts2(points,hor_rad,spr_rad):
        first_positive = np.array([i for i,p in enumerate(points) if p[0]>0])[0]
        print(first_positive)
        above2 = np.asarray([i+first_positive for i, p in enumerate(points[first_positive:]) if hor_rad+np.abs(p[0])>np.linalg.norm(p[1:])>hor_rad-np.abs(p[0]) and hor_rad-spr_rad+np.abs(p[0])<np.linalg.norm(p[1:])]) #and np.linalg.norm(p[1:])<hor_rad-spr_rad-np.abs(p[0])
        below2 = np.asarray([i for i, p in enumerate(points[:first_positive]) if hor_rad+np.abs(p[0])>np.linalg.norm(p[1:])>hor_rad-np.abs(p[0]) and hor_rad-spr_rad+np.abs(p[0])<np.linalg.norm(p[1:])])
        return above2, below2
