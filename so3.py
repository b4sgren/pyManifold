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
        if isinstance(R2, SO3): #Do I want to define this for other 3x3 matrices?
            return SO3(self.R @ R2.R)
        elif isinstance(R2, np.ndarray):
            if not R2.size == 3:
                raise ValueError("R2 needs to be a 1D array of length 3")
            else:
                return self.R @ R2 
        else:
            raise ValueError("Type not supported. Make sure R2 is an SO3 object or a numpy array")
    
    def __sub__(self, R2): #May add a vector to define as a box minus
        if isinstance(R2, SO3):
            return self.R - R2.R
        else:
            raise ValueError("Type not supported. Make sure R2 is an SO3 object")

    def inv(self):
        return SO3(self.arr.T)
    
    def transpose(self):
        return SO3(self.arr.T)

    @property 
    def R(self):
        return self.arr
    
    @classmethod 
    def fromRPY(cls, angles):
        phi = angles[0]
        theta = angles[1]
        psi = angles[2]
        
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
    def fromAxisAngle(cls, w): 
        theta = np.linalg.norm(w)
        skew_w = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

        arr = np.eye(3) + np.sin(theta) / theta * skew_w + (1 - np.cos(theta)) / (theta**2) * (skew_w @ skew_w)

        return cls(arr)

    @staticmethod 
    def log(R):
        theta = np.arccos((np.trace(R.arr) - 1)/2.0)
        if theta > 1e-3:
            return theta / (2.0 * np.sin(theta)) * (R - R.transpose()) # Define subtraction operator
        else: # Do taylor series expansion
            temp = 1/2.0 * (1 + theta**2 / 3.0 + 7 * theta**4 / 360)
            return temp * (R - R.transpose())
    
    @classmethod 
    def exp(cls, logR):
        w = cls.vee(logR) 
        theta = np.sqrt(w @ w)
        if theta > 1e-3:
            R = np.eye(3) + np.sin(theta)/theta * logR + (1 - np.cos(theta))/ (theta**2) * (logR @ logR)
        else: # Do taylor series expansion for small thetas
            stheta = 1 - theta**2 / 6.0 + theta**4 / 120.0 
            ctheta = 1/2.0 - theta**2 / 24.0 + theta**4 / 720
            R = np.eye(3) + stheta * logR + ctheta * (logR @ logR)

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