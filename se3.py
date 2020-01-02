import numpy as np
from IPython.core.debugger import Pdb

G = np.zeros((6,4,4))
G[0,0,3] = 1
G[1,1,3] = 1
G[2,2,3] = 1
G[3,1,2] = -1 #Sign may be switched for all from this line on down for passive rotations
G[3,2,1] = 1
G[4,0,2] = 1
G[4,2,0] = -1
G[5,0,1] = -1
G[5,1,0] = 1

class SE3:
    def __init__(self, t, *args):
        self.arr = np.eye(4)
        self.arr[:3, 3] = t

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
    
    @property 
    def R(self):
        return self.arr[:3,:3]
    
    @property
    def t(self):
        return self.arr[:3,3]

    @staticmethod
    def log(T): #Add an R and t property to get rotation and translation
        theta = np.arccos((np.trace(T.arr[:3,:3]) - 1)/2.0) 
        logR = theta / (2.0 * np.sin(theta)) * (T.R - T.R.T)

        A = np.sin(theta)/theta 
        B = (1 - np.cos(theta))/ (theta**2)

        V_inv = np.eye(3) - 0.5 * logR + 1/theta**2 * (1 - A/(2 * B)) * (logR @ logR)
        u = V_inv @ T.t

        logT = np.zeros((4,4))
        logT[:3,:3] = logR 
        logT[:3,3] = u 

        return logT
    
    @classmethod
    def exp(cls, logT):
        u = logT[:3,3]
        w = np.array([logT[2,1], logT[0,2], logT[1,0]])

        theta = np.sqrt(w @ w)
        A = np.sin(theta) / theta 
        B = (1 - np.cos(theta)) / (theta**2)
        C = (1 - A) / (theta**2)

        R = np.eye(3) + A * logT[:3,:3] + B * np.linalg.matrix_power(logT[:3,:3], 2)
        V = np.eye(3) + B * logT[:3,:3] + C * np.linalg.matrix_power(logT[:3,:3], 2)

        t = V @ u
        return cls(t, R)
    
    @staticmethod 
    def vee(logT):
        u = logT[:3,3]
        w = np.array([logT[2,1], logT[0,2], logT[1,0]])

        return np.concatenate((u,w))
    
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
        adj[:3,-3:] = tx @ R

        return adj