import numpy as np 

# J = np.array([[[0, 1], #Not sure I will use this since I have 3 complex numbers
#                [1, 0]],
#                [[0, complex(0, -1)],
#                [complex(0, 1), 0]],
#                [[1, 0],
#                [0, -1]]])

class Quaternion:
    def __init__(self, q):
        if isinstance(q, np.ndarray):
            if q.shape == (4,) or q.shape == (4,1) or q.shape == (1,4):
                self.arr = q.squeeze()
            else:
                raise ValueError("Input must be a numpy array of length 4")
        else:
            raise ValueError("Input must be a numpy array of length 4")
    
    @classmethod 
    def fromRotationMatrix(cls, R):
        if not R.shape == (3,3):
            raise ValueError("Input must be  3x3 numpy array")        

        delta = np.trace(R)
        if delta > 0:
            s = 2 * np.sqrt(delta + 1)
            q = np.array([s/4, 1/s*(R[1,2] - R[2,1]), 1/s * (R[2,0] - R[0,2]), 1/s * (R[0,1] - R[1,0])])
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2 * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
            q = np.array([1/s * (R[1,2] - R[2,1]), s/4.0, 1/s * (R[1,0] + R[0,1]), 1/s * (R[2,0] + R[0,2])])
        elif R[1,1] > R[2,2]:
            s = 2 * np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
            q = np.array([(R[2,0] - R[0,2])/2, 1/s * (R[1,0] + R[0,1]), s/4, 1/s * (R[2,1] + R[1,2])])
        else:
            s = 2 * np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
            q = np.array([1/s * (R[0,1] - R[1,0]), 1/s * (R[2,0] + R[0,2]), 1/s * (R[2,1] + R[1,2]), s/4])
        
        return cls(q)
    
    @classmethod
    def fromAxisAngle(cls, vec):
        if not vec.shape == (3,) and not vec.shape == (3,1) and not vec.shape == (1,3):
            raise ValueError("Input must be a numpy array of length 3")

        theta = np.linalg.norm(vec)
        v = (vec / theta).squeeze() #Make a unit vector
        st2 = np.sin(theta/2)
        q = np.array([np.cos(theta/2), v[0] * st2, v[1] * st2, v[2] * st2])
        return cls(q)
    
    @staticmethod 
    def hat(omega):
        if isinstance(omega, np.ndarray):
            if omega.shape == (3,) or omega.shape == (3,1) or omega.shape == (1,3):
                q = np.array([0, omega.item(0), omega.item(1), omega.item(2)])
            else:
                raise ValueError('Input must be an numpy array of length 3')
        else:
            raise ValueError('Input must be a numpy array of length 3')

        return q
    
    @staticmethod
    def vee(log_q):
        if isinstance(log_q, np.ndarray):
            if log_q.shape == (4,) or log_q.shape == (4,1) or log_q.shape == (1,4):
                log_q = log_q.squeeze()
                w = log_q[1:]
            else:
                raise ValueError('Input must be a numpy array of length 4')
        else:
            raise ValueError('Input must be a numpy array of length 4')

        return w