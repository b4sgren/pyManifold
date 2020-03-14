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


class SE3:
    def __init__(self, t, *args): #Add a from euler, and rot_vec
        self.arr = np.eye(4)
        if t.size == 16:
            self.arr = t 
        elif t.size == 3:
            self.arr[:3, 3] = t
        else:
            raise ValueError("T must be a 4x4 transformation matrix or a 3 vector for translation")

        if len(args) == 1: #Rotation passed in as a rotation matrix
            self.arr[:3,:3] = args[0]
        if len(args) == 3: #Rotation passed in as RPY angles in that order 
            phi = args[0]
            theta = args[1]
            psi = args[2]

            cps = np.cos(psi)
            sps = np.sin(psi)
            R1 = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])

            ct = np.cos(theta)
            st = np.sin(theta)
            R2 = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])

            cp = np.cos(phi)
            sp = np.sin(phi)
            R3 = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])

            self.arr[:3,:3] = R1 @ R2 @ R3
    
    def inv(self):
        return SE3(-self.R.T @ self.t, self.R.T)
    
    def __mul__(self, T):
        if isinstance(T, SE3):
            temp = self.arr @ T.arr
            return SE3(temp[:3,3], temp[:3,:3])
        elif isinstance(T, np.ndarray):
            if not T.size == 3:
                raise ValueError("T is the incorrect shape. T must be a 1D array with length 3")
            else:
                temp = self.arr @ np.hstack((T, [1])) 
                return temp[:-1]
        else:
            raise ValueError("Type not supported. T must be an SE3 object or an numpy array with length 3")
    
    @property 
    def R(self):
        return self.arr[:3,:3]
    
    @property
    def t(self):
        return self.arr[:3,3]
    
    @classmethod 
    def fromRPY(cls, t, angles): #Test this
        if not angles.size == 3:
            raise ValueError("To use fromRPY the input must be a numpy array of length 3")
        if not t.size == 3:
            raise ValueError("The translation vector must be a numppy array of length 3")

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

        return cls(t, R1 @ R2 @ R3)
    
    @classmethod 
    def fromAxisAngle(cls, t, w): #Need to test this
        if not t.size == 3:
            raise ValueError("The translation vector must be a numpy array of length 3")
        if not w.size == 3:
            raise ValueError("The axis angle vector must be a numpy array of length 3. The norm of the vector is the angle of rotation")

        theta = np.linalg.norm(w)
        skew_w = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

        arr = np.eye(3) + np.sin(theta) / theta * skew_w + (1 - np.cos(theta)) / (theta**2) * (skew_w @ skew_w) #Should consider Taylor series for this
        return cls(t, arr)

    @staticmethod
    def log(T):  #Do taylor series expansion
        theta = np.arccos((np.trace(T.arr[:3,:3]) - 1)/2.0) 
        logR = theta / (2.0 * np.sin(theta)) * (T.R - T.R.T)

        if theta > 1e-3 and np.abs(np.abs(theta) - np.pi) > 1e-3:
            A = np.sin(theta)/theta 
            B = (1 - np.cos(theta))/ (theta**2)
        else:
            A = 1.0 - theta**2/6.0 + theta**4/120.0
            B = 0.5 - theta**2/24.0 + theta**4/720.0

        V_inv = np.eye(3) - 0.5 * logR + 1/theta**2 * (1 - A/(2 * B)) * (logR @ logR)
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
        u = logT[:3,3]
        w = np.array([logT[2,1], logT[0,2], logT[1,0]])

        theta = np.sqrt(w @ w)
        if theta > 1e-3:
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
        return cls(t, R)
    
    @classmethod 
    def Exp(cls, arr): #One call to go from a vector to Transformation matrix
        logT = cls.hat(arr)
        return cls.exp(logT)
    
    @staticmethod 
    def vee(logT):
        u = logT[:3,3]
        w = np.array([logT[2,1], logT[0,2], logT[1,0]])

        return np.concatenate((w, u))
    
    @staticmethod 
    def hat(arr):
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