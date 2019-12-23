import numpy as np 

#G might change if the R part of arr changes
G = np.array([[[0, 0, 1],
                [0, 0, 0],
                [0, 0, 0]],
               [[0, 0, 0],
                [0, 0, 1],
                [0, 0, 0]],
               [[0, -1, 0],
                [1, 0, 0], 
                [0, 0, 0]]])
class SE2:
    def __init__(self, R, t):
        self.arr = np.eye(3)
        self.arr[:2,:2] = R 
        self.arr[:2,2] = t
    
    def inv(self):
        temp = np.zeros_like(self.arr)
        return SE2(self.arr[:2,:2].T, -self.arr[:2,:2].T @ self.arr[:2,2])

    def __mul__(self, T2):
        temp = self.arr @ T2.arr
        return SE2(temp[:2,:2], temp[:2,2])
        