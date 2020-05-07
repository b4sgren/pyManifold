import numpy as np 

from IPython.core.debugger import Pdb

class Quaternion:
    def __init__(self, q):
        if isinstance(q, np.ndarray):
            if q.shape == (4,) or q.shape == (4,1) or q.shape == (1,4):
                self.arr = q.squeeze()
                if self.arr[0] < 0:
                    self.arr *= -1
            else:
                raise ValueError("Input must be a numpy array of length 4")
        else:
            raise ValueError("Input must be a numpy array of length 4")
    
    @property
    def qw(self):
        return self.arr[0]
    
    @property
    def qx(self):
        return self.arr[1]
    
    @property 
    def qy(self):
        return self.arr[2]
    
    @property 
    def qz(self):
        return self.arr[3]
    
    @property
    def qv(self):
        return self.arr[1:]
    
    @property 
    def q(self):
        return self.arr
    
    @property 
    def R(self): #The problem is in here I believe
        return (2 * self.qw**2 - 1) * np.eye(3) - 2 * self.qw * self.skew() + 2 * np.outer(self.qv, self.qv)
    
    def __mul__(self, q):
        return self.otimes(q)
    
    def otimes(self, q):
        Q = np.block([[self.qw, -self.qv], [self.qv[:,None], self.qw * np.eye(3) + self.skew()]]) #Typo in Jame's stuff. See QUat for Err State KF
        return Quaternion(Q @ q.q)
    
    def skew(self):
        qv = self.qv
        return np.array([[0, -qv[2], qv[1]], [qv[2], 0, -qv[0]], [-qv[1], qv[0], 0]])
    
    def inv(self):
        return Quaternion(np.array([self.qw, -self.qx, -self.qy, -self.qz]))
    
    @classmethod
    def random(cls): #Method found at planning.cs.uiuc.edu/node198.html (SO how to generate a random quaternion quickly)
        u = np.random.uniform(0.0, 1.0, size=3)
        qw = np.sin(2 * np.pi * u[1]) * np.sqrt(1 - u[0])
        q1 = np.cos(2 * np.pi * u[1]) * np.sqrt(1 - u[0])
        q2 = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
        q3 = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
        return Quaternion(np.array([qw, q1, q2, q3]))
    
    @classmethod 
    def fromRotationMatrix(cls, R):
        d = np.trace(R)
        if d > 0:
            s = 2 * np.sqrt(d + 1)
            q = np.array([s/4, 1/s * (R[1,2] - R[2,1]), 1/s * (R[2,0] - R[0,2]), 1/s * (R[0,1] - R[1,0])])
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2 * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
            q = np.array([1/s * (R[1,2] - R[2,1]), s/4, 1/s * (R[1,0] + R[0,1]), 1/s * (R[2,0] + R[0,2])])
        elif R[1,1] > R[2,2]:
            s = 2 * np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
            q = np.array([1/s * (R[2,0] - R[0,2]), 1/s * (R[1,0] + R[0,1]), s/4, 1/s * (R[2,1] + R[1,2])])
        else:
            s = 2 * np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
            q = np.array([1/s * (R[0,1] - R[1,0]), 1/s * (R[2,0] + R[0,2]), 1/s * (R[2,1] + R[1,2]), s/4])
        
        return Quaternion(q)