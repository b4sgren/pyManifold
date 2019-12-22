import numpy as np 

G = np.array([[0, -1], [1, 0]])
class SO2:
    def __init__(self, theta):
        self.arr = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]]) #SHould the - sign be switched for our application? 
    
    @classmethod
    def exp(cls, theta_x):
        theta = theta_x[1,0]
        return cls(theta)
    
    @staticmethod
    def log(R): 
        # theta = np.arccos(R.arr[0,0])
        theta = np.arctan2(R.arr[1,0], R.arr[0,0])
        return G * theta

    def vee(self, v):
        debug = 1

    def hat(self, v):
        debug = 1
    
    def boxplus(self, R2):
        debug = 1 #Make this a classmethod?

    def boxminux(self, R2):
        debug = 1 #staticmethod?