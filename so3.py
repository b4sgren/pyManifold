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
    def __init__(self, args1, *args): # Add capability to check if phi is an 3x3 matrix instead of an angle
        if len(args) == 0: #args1 is already a 3x3 rotation matrix
            self.arr = args1 
        elif len(args) == 2: #Arguments passed in are RPY angles in that order (roll, pitch, yaw)
            phi = args1
            theta = args[0]
            psi = args[1]

            #May need to switch the signs on all these rotation matrices to get them to be passive rotations
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
    
    @staticmethod 
    def log(R):
        theta = np.arccos((np.trace(R.arr) - 1)/2.0)
        return theta / (2.0 * np.sin(theta)) * (R.arr - R.arr.T)