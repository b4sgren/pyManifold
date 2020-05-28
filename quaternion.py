import numpy as np 

def skew(qv):
    return np.array([[0, -qv[2], qv[1]], [qv[2], 0, -qv[0]], [-qv[1], qv[0], 0]])

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
    def R(self): 
        return (2 * self.qw**2 - 1) * np.eye(3) - 2 * self.qw * skew(self.qv) + 2 * np.outer(self.qv, self.qv)
    
    @property 
    def Adj(self):
        return self.R.T
    
    def __mul__(self, q): #may need to define this for the reverse order
        return self.otimes(q)
    
    def otimes(self, q):
        Q = np.block([[self.qw, -self.qv], [self.qv[:,None], self.qw * np.eye(3) + self.skew()]]) #Typo in Jame's stuff. See QUat for Err State KF
        return Quaternion(Q @ q.q)
    
    def skew(self):
        qv = self.qv
        return np.array([[0, -qv[2], qv[1]], [qv[2], 0, -qv[0]], [-qv[1], qv[0], 0]])
    
    def inv(self):
        return Quaternion(np.array([self.qw, -self.qx, -self.qy, -self.qz]))
    
    def rota(self, v):
        qw = self.qw 
        qv = self.qv

        t = 2 * skew(v) @ qv
        return v + qw * t + skew(t) @ qv
    
    def rotp(self, v):
        qw = self.qw
        qv = self.qv 

        t = 2 * skew(v) @ qv 
        return v - qw * t + skew(t) @ qv
    
    def normalize(self):
        self.arr = self.q / self.norm()
    
    def norm(self):
        return np.linalg.norm(self.q)
    
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
    
    @classmethod 
    def fromRPY(cls, rpy):
        phi = rpy[0]
        theta = rpy[1]
        psi = rpy[2]

        cp = np.cos(phi/2)
        sp = np.sin(phi/2)
        ct = np.cos(theta/2)
        st = np.sin(theta/2)
        cpsi = np.cos(psi/2)
        spsi = np.sin(psi/2)

        qw = cpsi * ct * cp + spsi * st * sp  #The sign on the last three are opposite the UAV book b/c we are generating an active quaternion
        qx = -cpsi * ct * sp + spsi * st * cp 
        qy = -cpsi * st * cp - spsi * ct * sp 
        qz = -spsi * ct * cp + cpsi * st * sp 
        return cls(np.array([qw, qx, qy, qz]))
    
    @classmethod 
    def fromAxisAngle(cls, vec):
        return cls.Exp(vec)
    
    @staticmethod
    def hat(w):
        return np.array([0, *w])
    
    @staticmethod 
    def vee(W):
        return W[1:]
    
    @staticmethod 
    def log(q): #TODO: Taylor series expansion
        qw = q.qw 
        qv = q.qv 
        w = 2 * np.arctan(np.linalg.norm(qv)/qw) * qv/np.linalg.norm(qv)
        return np.array([0, *w]) #I have never seen anything that says this is negative but when I compare with a Rotation matrix I get the negative values of a matrix log
    
    @staticmethod 
    def Log(q):
        W = Quaternion.log(q)
        return Quaternion.vee(W)
    
    @classmethod
    def exp(cls, W): 
        vec = W[1:]
        theta = np.linalg.norm(vec)
        v = vec / theta

        qw = np.cos(theta/2)
        qv = v * np.sin(theta/2)
        return cls(np.array([qw, *qv]))
    
    @staticmethod 
    def Exp(w):
        W = Quaternion.hat(w)
        return Quaternion.exp(W)
    
    #Jacobians