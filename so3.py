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
    def __init__(self, args1, *args): # NOTE: Would it be better to have this be a scipy rotation object? Easily convert to axis-angle, rot. matrix, euler angles and quaternion this way
        if len(args) == 0: #args1 a 3x3 rotation matrix
            self.arr = args1 
        elif len(args) == 2: #args1 is roll, args has pitch and yaw
            phi = args1
            theta = args[0]
            psi = args[1]

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

            self.arr = R1 @ R2 @ R3
    
    def __mul__(self, R2):
        return SO3(self.arr @ R2.arr)

    @property 
    def R(self):
        return self.arr
    
    @staticmethod 
    def log(R):
        theta = np.arccos((np.trace(R.arr) - 1)/2.0)
        return theta / (2.0 * np.sin(theta)) * (R.arr - R.arr.T)
    
    @classmethod 
    def exp(cls, logR):
        w = np.array([logR[2,1], logR[0,2], logR[1,0]])
        theta = np.sqrt(w @ w)
        temp = np.eye(3) + np.sin(theta)/theta * logR + (1 - np.cos(theta))/ (theta**2) * (logR @ logR)
        return cls(temp)
    
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