import numpy as np 

class SO2:
    def __init__(self, theta):
        self.R = np.array([[np.cos(theta), np.sin(theta)],
                            [-np.sin(theta), np.cos(theta)]])
        
    def exp(self, v):
        debug = 1
    
    def log(self, R2):
        debug = 1

    def vee(self, v):
        debug = 1

    def hat(self, v):
        debug = 1
    
    def boxplus(self, R2):
        debug = 1 #Make this a classmethod?

    def boxminux(self, R2):
        debug = 1 #staticmethod?