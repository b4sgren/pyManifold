import numpy as np 

G = np.array([[[0, -1, 0],
                [1, 0, 0],
                [0, 0, 0]],
               [[0, 0, 1],
                [0, 0, 0],
                [0, 0, 0]],
               [[0, 0, 0],
                [0, 0, 1], 
                [0, 0, 0]]])
class SE2:
    def __init__(self, R, t):
        self.arr = np.eye(3)
        self.arr[:2,:2] = R 
        self.arr[:2,2] = t
    
    def inv(self):
        return SE2(self.arr[:2,:2].T, -self.arr[:2,:2].T @ self.arr[:2,2])

    def __mul__(self, T2): 
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
        J = np.array([[0, 1], [-1, 0]])
        adj = np.zeros((3,3))
        adj[0,0] = 1
        adj[1:,0] = J @ self.t 
        adj[1:,1:] = self.R

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
    def log(T): #Implement taylor series expansion
        theta = np.arctan2(T.arr[1,0], T.arr[0,0])
        t = T.t

        if np.abs(theta) > 1e-3:
            A = np.sin(theta)/theta
            B = (1 - np.cos(theta))/theta 
        else: 
            A = 1 - theta**2 / 6.0 + theta**4 / 120.0
            B = theta/2.0 - theta**3 / 24.0 + theta**5/720.0
        normalizer = 1 / (A**2 + B**2)
        V_inv = normalizer * np.array([[A, B], [-B, A]])

        logT = np.zeros((3,3))
        logT[:2,2] = V_inv @ t 
        logT[0,1] = -theta
        logT[1,0] = theta

        return logT 
    
    @classmethod 
    def Log(cls, T):
        logT = cls.log(T)
        return cls.vee(logT)
    
    @classmethod
    def exp(cls, X): #Taylor series expansion
        theta = X[1,0]
        ct = np.cos(theta)
        st = np.sin(theta)

        V = 1/theta * np.array([[st, ct-1], [1 - ct, st]]) 
        t = V @ X[:2,2]
        
        return cls.fromAngle(theta, t)
    
    @classmethod 
    def Exp(cls, vec):
        logR = cls.hat(vec)
        return cls.exp(logR)
    
    @staticmethod
    def vee(X):
        arr = np.zeros(3)
        arr[1:] = X[:2,2]
        arr[0] = X[1,0]

        return arr
    
    @staticmethod
    def hat(arr): 
        return np.sum(G * arr[:,None, None], axis=0)