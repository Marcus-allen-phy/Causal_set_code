"""
=========================================
Base Simulation (:mod:`base_simulation`)
=========================================
"""
from methods import *
from parallel import *

class BaseSimulation(ABC):
    """Base Simulation Class.
    
    Base class containing all basic abstract and class methods for general simulations.
    """
    
    @abstractmethod
    def __init__(self, dim: int = 2, num_points: int = 500):
        self.dim = dim
        self.num_points = num_points
        self.coords = None
        
    def get_coordinates(self):
        return self.coords
    
    def number_points(self):
        if self.poisson:
            self.n = np.random.poisson(self.rho*self.volume)
        else:
            self.n = round(self.rho*self.volume)

    def plot(self, ax = None):
        if self.dim > 3:
            raise ValueError('Can only plot up to dimension 3')
        elif self.dim == 2:
            if ax is None:
                fig, ax = plt.subplots(1,1,figsize=(5,5))
            ax.plot(self.coords[0], self.coords[1], marker='o', linestyle='', color='lightskyblue')
        else:
            if ax is None:
                fig = plt.figure(figsize=(5,5)) 
                ax = fig.add_subplot(111, projection='3d')
            ax.scatter3D(self.coords[0], self.coords[1], self.coords[2], marker='o', color='lightskyblue')

    