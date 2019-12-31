import numpy as np

G = np.zeros((6,4,4))
G[0,0,3] = 1
G[1,1,3] = 1
G[2,2,3] = 1
G[3,2,3] = -1 #Sign may be switched for all from this line on down for passive rotations
G[3,3,2] = 1
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