"""
=============================================================
Dynamical Shell Horizon Simulation (:mod:`dynamical_shell`)
=============================================================
"""

from base_simulation import *

class DynamicalShellHorizon(BaseSimulation):
    """Dynamical Shell Horizon Simulation class.
    
    TODO describe Dynamical Shell Horizon
    """
    
    def __init__(self, dim: int = 2, rho: int = 500, height: int = 3, proportion: int = 1, poisson: bool = True):
        self.dim = dim
        if self.dim<2:
            raise ValueError('Dimension must be at least 2')
        self.rho = rho
        self.height = height
        self.radius = height/3
        self.circ = (self.radius)*2*np.pi
        self.proportion = proportion
        self.poisson = poisson
        if self.dim  == 2:
            self.volume = 2*((self.height)**2 - (self.height*(1-self.proportion))*self.height)
        if self.dim == 3:
            self.volume = 2*(np.pi/3)*(self.height**3 - (self.height*(1-self.proportion))*(self.height)**2)
        
    
    def coordinates(self):
        maxheight = self.height**(self.dim)
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
    def cuts2(points,radius):
        first_positive = np.array([i for i,p in enumerate(points) if p[0]>0])[0]
        above2 = np.asarray([i+first_positive for i, p in enumerate(points[first_positive:]) if radius+np.abs(p[0])>np.linalg.norm(p[1:])>radius-np.abs(p[0])])
        below2 = np.asarray([i for i, p in enumerate(points[:first_positive]) if radius+np.abs(p[0])>np.linalg.norm(p[1:])>radius-np.abs(p[0])])
        return above2, below2