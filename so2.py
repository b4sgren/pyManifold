import numpy as np 

G = np.array([[0, -1], [1, 0]])
class SO2:
    def __init__(self, theta):
        self.arr = np.array([[np.cos(theta), np.sin(theta)],
                            [-np.sin(theta), np.cos(theta)]])
    
    def exp(self, v):
        debug = 1
    
    @staticmethod
    def log(R):
        theta = np.arccos(R[0,0])
        return G * theta

    def vee(self, v):
        debug = 1

    def hat(self, v):
        debug = 1
    
    def boxplus(self, R2):
        debug = 1 #Make this a classmethod?

    def boxminux(self, R2):
        debug = 1 #staticmethod?