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