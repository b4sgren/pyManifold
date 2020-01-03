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
        return SE2(self.arr[:2,:2].T, -self.arr[:2,:2].T @ self.arr[:2,2])

    def __mul__(self, T2): # May need to check if this is an SE2 object or a point to be transformed. 
        if isinstance(T2, SE2):
            temp = self.arr @ T2.arr
            return SE2(temp[:2,:2], temp[:2,2])
        elif isinstance(T2, np.ndarray):
            if not T2.size == 2:
                raise ValueError("Array is to long. Make sure the array is a 1D array with size 2")
            else:
                return self.R @ T2 + self.t
        else:
            raise ValueError("Type not supported. Make sure the type is an SE2 object")
    

    @property 
    def Adj(self):
        adj = np.eye(3)
        adj[:2, :2] = self.arr[:2, :2]
        adj[0, 2] = self.arr[1, 2]
        adj[1, 2] = -self.arr[0, 2]

        return adj
    
    @property 
    def R(self):
        return self.arr[:2,:2]
    
    @property 
    def t(self):
        return self.arr[:2,2]
    
    @classmethod
    def fromAngle(cls, theta, t):
        ct = np.cos(theta)
        st = np.sin(theta)
        R = np.array([[ct, -st], [st, ct]])

        return cls(R, t)
    
    @staticmethod
    def log(T):
        theta = np.arctan2(T.arr[1,0], T.arr[0,0])
        t = T.t

        A = np.sin(theta)/theta
        B = (1 - np.cos(theta))/theta 
        normalizer = 1 / (A**2 + B**2)
        V_inv = normalizer * np.array([[A, B], [-B, A]])

        logT = np.zeros((3,3))
        logT[:2,2] = V_inv @ t 
        logT[0,1] = -theta
        logT[1,0] = theta

        return logT 
    
    @classmethod
    def exp(cls, X):
        theta = X[1,0]
        ct = np.cos(theta)
        st = np.sin(theta)

        V = 1/theta * np.array([[st, ct-1], [1 - ct, st]])
        t = V @ X[:2,2]
        
        return cls.fromAngle(theta, t)
    
    @staticmethod
    def vee(X):
        arr = np.zeros(3)
        arr[:2] = X[:2,2]
        arr[2] = X[1,0]

        return arr
    
    @staticmethod
    def hat(arr): 
        return np.sum(G * arr[:,None, None], axis=0)