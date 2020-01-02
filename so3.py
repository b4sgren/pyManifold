import numpy as np 

G = np.array([[[0, 0, 0],
                [0, 0, -1],
                [0, 1, 0]],
               [[0, 0, 1],
                [0, 0, 0],
                [-1, 0, 0]],
               [[0, -1, 0],
                [1, 0, 0],
                [0, 0, 0]]])

class SO3:
    def __init__(self, R): 
        self.arr = R 
    
    def __mul__(self, R2):
        return SO3(self.arr @ R2.arr)

    def inv(self):
        return SO3(self.arr.T)

    @property 
    def R(self):
        return self.arr
    
    @classmethod 
    def fromRPY(cls, angles):
        phi = angles[0]
        theta = angles[1]
        psi = angles[2]
        
        #NOTE: May need to switch the signs on all these rotation matrices to get them to be passive rotations
        cps = np.cos(psi)
        sps = np.sin(psi)
        R1 = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])

        ct = np.cos(theta)
        st = np.sin(theta)
        R2 = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])

        cp = np.cos(phi)
        sp = np.sin(phi)
        R3 = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])

        return cls(R1 @ R2 @ R3)
    
    @classmethod 
    def fromAxisAngle(cls, vec):
        debug = 1

    @classmethod #Not sure that I want this one. I will have a separate quaternion class that I want to implement
    def fromQuaternion(cls, q):
        debug = 1
    
    @staticmethod 
    def log(R):
        theta = np.arccos((np.trace(R.arr) - 1)/2.0)
        return theta / (2.0 * np.sin(theta)) * (R.arr - R.arr.T)
    
    @classmethod 
    def exp(cls, logR):
        w = np.array([logR[2,1], logR[0,2], logR[1,0]])
        theta = np.sqrt(w @ w)
        R = np.eye(3) + np.sin(theta)/theta * logR + (1 - np.cos(theta))/ (theta**2) * (logR @ logR)
        return cls(R)
    
    @staticmethod 
    def vee(logR):
        omega = np.array([logR[2,1], logR[0,2], logR[1,0]])
        return omega
    
    @staticmethod
    def hat(omega):
        return (G @ omega).squeeze()
    
    @property 
    def Adj(self): #Need to test this still
        return self.arr