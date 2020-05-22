import numpy as np

G = np.zeros((6,4,4))
G[0,1,2] = -1 
G[0,2,1] = 1
G[1,0,2] = 1
G[1,2,0] = -1
G[2,0,1] = -1
G[2,1,0] = 1
G[3,0,3] = 1
G[4,1,3] = 1
G[5,2,3] = 1

#Jacobians, and overload plus and minus for box plus and box minus

class SE3:
    def __init__(self, T): 
        assert T.shape == (4,4)
        self.arr = T
    
    def inv(self):
        R_inv = self.R.T 
        t_inv = -R_inv @ self.t
        return SE3.fromRotationMatrix(t_inv, R_inv)
    
    def __mul__(self, T):
        assert isinstance(T, SE3)
        temp = self.arr @ T.arr
        return SE3(temp)
        # elif isinstance(T, np.ndarray):
            # if not T.size == 3:
                # raise ValueError("T is the incorrect shape. T must be a 1D array with length 3")
            # else:
                # temp = self.arr @ np.hstack((T, [1])) 
                # return temp[:-1]
        # else:
            # raise ValueError("Type not supported. T must be an SE3 object or an numpy array with length 3")
    
    def transa(self, vec):
        assert vec.size == 3
        v = np.array([*vec, 1])
        vp = self.T @ v 
        return vp[:3]
    
    def transp(self, vec):
        assert vec.size == 3
        v = np.array([*vec, 1])
        vp = self.inv().T @ v 
        return vp[:3]
    
    @property 
    def R(self):
        return self.arr[:3,:3]
    
    @property
    def t(self):
        return self.arr[:3,3]
    
    @property 
    def T(self):
        return self.arr
    
    @classmethod 
    def fromRotationMatrix(cls, t, R):
        assert t.size == 3
        assert R.shape == (3,3)
        T = np.block([[R, t[:,None]], [np.zeros(3), 1]])
        return cls(T)
    
    @classmethod 
    def fromRPY(cls, t, angles): 
        assert angles.size == 3
        assert t.size == 3

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

        return cls.fromRotationMatrix(t, R1 @ R2 @ R3)
    
    @classmethod 
    def fromAxisAngle(cls, t, w): 
        assert t.size == 3
        assert w.size == 3

        theta = np.linalg.norm(w)
        skew_w = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

        if theta > 1e-3:
            A = np.sin(theta)/ theta 
            B = (1 - np.cos(theta)) / (theta**2)
        else:
            A = 1 - theta**2 / 6.0 + theta**4 / 120.0
            B = 0.5 - theta**2 / 24.0 + theta**4 / 720.0

        arr = np.eye(3) + A * skew_w + B * (skew_w @ skew_w) 
        return cls.fromRotationMatrix(t, arr)

    @staticmethod
    def log(T):  
        assert isinstance(T, SE3)

        theta = np.arccos((np.trace(T.arr[:3,:3]) - 1)/2.0) 
        if np.abs(theta) < 1e-3:
            temp = 1/2.0 * (1 + theta**2 / 6.0 + 7 * theta**4 / 360) 
            logR =  temp * (T.R - T.R.T)
        elif np.abs(np.abs(theta) - np.pi) < 1e-3:
            temp = - np.pi/(theta - np.pi) - 1 - np.pi/6 * (theta - np.pi) - (theta - np.pi)**2/6 - 7*np.pi/360 * (theta - np.pi)**3 - 7/360.0 * (theta - np.pi)**4
            logR =  temp/2.0 * (T.R - T.R.T)
        else:
            logR = theta / (2.0 * np.sin(theta)) * (T.R - T.R.T) 

        if np.abs(theta) > 1e-3: 
            temp = np.sin(theta) / (theta * (1 - np.cos(theta)))
        else:
            temp = 2/theta**2 - 1/6.0 - theta**2/360.0 - theta**4/15120.0

        V_inv = np.eye(3) - 0.5 * logR + (1/theta**2 - temp/2) * (logR @ logR)
        u = V_inv @ T.t

        logT = np.zeros((4,4))
        logT[:3,:3] = logR 
        logT[:3,3] = u 

        return logT
    
    @classmethod #One call to go from Transformation matrix to vector
    def Log(cls, T):
        logT = cls.log(T)
        return cls.vee(logT)
    
    @classmethod
    def exp(cls, logT):
        assert logT.shape == (4,4)

        u = logT[:3,3]
        w = np.array([logT[2,1], logT[0,2], logT[1,0]])

        theta = np.sqrt(w @ w)
        if np.abs(theta) > 1e-3:
            A = np.sin(theta) / theta  
            B = (1 - np.cos(theta)) / (theta**2)
            C = (1 - A) / (theta**2)
        else: #Taylor series expansion
            A = 1.0 - theta**2/6.0 + theta**4/120.0 
            B = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
            C = 1/6.0 - theta**2/120.0 + theta**4 / 5040.0

        R = np.eye(3) + A * logT[:3,:3] + B * np.linalg.matrix_power(logT[:3,:3], 2)
        V = np.eye(3) + B * logT[:3,:3] + C * np.linalg.matrix_power(logT[:3,:3], 2)

        t = V @ u
        return cls.fromRotationMatrix(t, R)
    
    @classmethod 
    def Exp(cls, arr): #One call to go from a vector to Transformation matrix
        logT = cls.hat(arr)
        return cls.exp(logT)
    
    @staticmethod 
    def vee(logT):
        assert logT.shape == (4,4)
        u = logT[:3,3]
        w = np.array([logT[2,1], logT[0,2], logT[1,0]])

        return np.concatenate((w, u))
    
    @staticmethod 
    def hat(arr):
        assert arr.size == 6
        return np.sum(G * arr[:,None, None], axis=0)

    @property
    def Adj(self):
        R = self.R
        t = self.t

        adj = np.zeros((6,6))
        adj[:3,:3] = R 
        adj[-3:,-3:] = R 

        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        adj[3:,:3] = tx @ R

        return adj
    
    #Left and right jacobian stuff goes here