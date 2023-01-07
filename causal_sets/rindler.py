"""
===========================================
Rindler Horizon Simulation (:mod:`rindler`)
===========================================
"""

from base_simulation import *


class RindlerHorizon(BaseSimulation):
    """Rindler Horizon Simulation class.
    
    TODO describe Rindler Horizon
    """
    
    def __init__(self, dim: int = 2, rho: int = 500, circ: int = 2, height: int = 1, proportion: int = 1, poisson: bool = True):
        self.dim = dim
        if self.dim<2:
            raise ValueError('Dimension must be at least 2')
        self.rho = rho
        self.circ = circ
        self.height = height
        self.proportion = proportion
        self.poisson = poisson
        self.volume = 2*((self.height)**2 - (self.height*(1-self.proportion))*self.height)*(self.circ)**(self.dim-2)
    
    
    def coordinates(self):
        self.dim = 2
        maxheight = self.height**(self.dim)
        h = (np.random.uniform((((maxheight**(1/self.dim))-(self.height*(self.proportion)))**self.dim),maxheight,self.n)**(1/self.dim))
        t = (maxheight**(1/self.dim)-h)*np.random.choice(np.array([-1, 1]), self.n)
        r = h*(np.random.uniform(0,1,self.n)**(1/(self.dim-1)))
        x_coords = r*np.random.choice(np.array([-1, 1]), self.n)
        self.dim = 3
        other = np.random.uniform(0,self.circ,(self.dim-2,self.n))
        if self.dim == 2:
            coords = np.vstack((t,x_coords))
        if self.dim ==3:
            coords = np.vstack((t,other[0],x_coords))
        if self.dim ==4:
            coords = np.vstack((t,other[0], other[1],x_coords))
        self.coords = coords[:,np.argsort(coords[0])]
        self.points = self.coords.T
        
    
    @staticmethod
    @njit(fastmath=True)
    def cuts4(points):
        above4 = np.asarray([i for i, p in enumerate(points) if p[0]>p[-1] and p[0]>-p[-1]])
        below4 = np.asarray([i for i, p in enumerate(points) if p[0]<p[-1] and p[0]< -p[-1]])
        left4 = np.asarray([i for i, p in enumerate(points) if p[0]>p[-1] and p[0]<-p[-1]])
        right4 = np.asarray([i for i, p in enumerate(points) if p[0]<p[-1] and p[0]> -p[-1]])
        return above4,below4,left4,right4
    

