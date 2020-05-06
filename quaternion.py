import numpy as np 

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
    def qv(self):
        return self.arr[1:]
    
    @property 
    def q(self):
        return self.arr
    
    def __mul__(self, q):
        return self.otimes(q)
    
    def otimes(self, q):
        Q = np.block([[-self.qw, -self.qv], [self.qv[:,None], self.qw * np.eye(3) + self.skew()]])
        return Quaternion(Q @ q.q)
    
    def skew(self):
        qv = self.qv
        return np.array([[0, -qv[2], qv[1]], [qv[2], 0, -qv[0]], [-qv[1], qv[0], 0]])
    
    @classmethod
    def random(cls): #Method found at planning.cs.uiuc.edu/node198.html (SO how to generate a random quaternion quickly)
        u = np.random.uniform(0.0, 1.0, size=3)
        qw = np.sin(2 * np.pi * u[1]) * np.sqrt(1 - u[0])
        q1 = np.cos(2 * np.pi * u[1]) * np.sqrt(1 - u[0])
        q2 = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
        q3 = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
        return Quaternion(np.array([qw, q1, q2, q3]))