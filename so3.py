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
        if R.shape[0] ==3 and R.shape[1] == 3:
            self.arr = R 
        else:
            raise ValueError("Input is a 3x3 numpy array. Otherwise use fromRPY or FromAxisAngle")
    
    def __mul__(self, R2): #I think that I will redefine the rotation on a vector as a separate function
        if isinstance(R2, SO3): 
            return SO3(self.R @ R2.R)
        elif isinstance(R2, np.ndarray):
            if not R2.shape[0] == 3:
                raise ValueError("R2 needs to be a 1D array of length 3")
            else:
                return self.R @ R2 
        else:
            raise ValueError("Type not supported. Make sure R2 is an SO3 object or a numpy array")
    
    def __sub__(self, R2): 
        if isinstance(R2, SO3):
            return self.R - R2.R #override this to do a box minus type thing
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

        if np.abs(theta) > 1e-3:
            A = np.sin(theta) / theta 
            B = (1 - np.cos(theta)) / (theta**2)
        else:
            A = 1.0 - theta**2 / 6.0 + theta**4 / 120.0
            B = 0.5 - theta**2 / 24.0 + theta**4/720.0

        arr = np.eye(3) + A * skew_w + B * (skew_w @ skew_w) #Do taylor series expansion

        return cls(arr)
    
    @classmethod 
    def fromQuaternion(cls, q):
        qw = q[0]
        qv = q[1:]
        qv_skew = np.array([[0, -qv[2], qv[1]], [qv[2], 0, -qv[0]], [-qv[1], qv[0], 0]])

        R = (2 * qw**2 - 1) * np.eye(3) - 2 * qw * qv_skew + 2 * np.outer(qv, qv)
        return cls(R) 
    
    @classmethod 
    def random(cls):
        x = np.random.uniform(0, 1, size=3)
        psi = 2 * np.pi * x[0]
        R = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        v = np.array([np.cos(2 * np.pi * x[1]) * np.sqrt(x[2]),
                     np.sin(2 * np.pi * x[1]) * np.sqrt(x[2]),
                     np.sqrt(1 - x[2])])
        H = np.eye(3) - 2 * np.outer(v, v)
        return cls(-H @ R)

    @staticmethod 
    def log(R):
        theta = np.arccos((np.trace(R.arr) - 1)/2.0)
        if np.abs(theta) < 1e-3: # Do taylor series expansion
            temp = 1/2.0 * (1 + theta**2 / 6.0 + 7 * theta**4 / 360) 
            return temp * (R - R.transpose())
        elif np.abs(np.abs(theta) - np.pi) < 1e-3:
            temp = - np.pi/(theta - np.pi) - 1 - np.pi/6 * (theta - np.pi) - (theta - np.pi)**2/6 - 7*np.pi/360 * (theta - np.pi)**3 - 7/360.0 * (theta - np.pi)**4
            return temp/2.0 * (R - R.transpose())
        else:
            return theta / (2.0 * np.sin(theta)) * (R - R.transpose()) 
    
    @classmethod 
    def Log(cls, R): #easy call to go straight to a vector
        logR = cls.log(R)
        return cls.vee(logR)
    
    @classmethod 
    def exp(cls, logR):
        w = cls.vee(logR) 
        theta = np.sqrt(w @ w)
        if np.abs(theta) > 1e-3:
            R = np.eye(3) + np.sin(theta)/theta * logR + (1 - np.cos(theta))/ (theta**2) * (logR @ logR)
        else: # Do taylor series expansion for small thetas
            stheta = 1 - theta**2 / 6.0 + theta**4 / 120.0 
            ctheta = 1/2.0 - theta**2 / 24.0 + theta**4 / 720
            R = np.eye(3) + stheta * logR + ctheta * (logR @ logR)

        return cls(R)
    
    @classmethod 
    def Exp(cls, w): # one call to go straight from vector to SO3 object
        logR = cls.hat(w)
        R = cls.exp(logR)
        return R
    
    @staticmethod 
    def vee(logR):
        omega = np.array([logR[2,1], logR[0,2], logR[1,0]])
        return omega
    
    @staticmethod
    def hat(omega):
        return (G @ omega).squeeze()
    
    @property 
    def Adj(self): 
        return self.arr
    
    #Left and right jacobians go here. Read about them first