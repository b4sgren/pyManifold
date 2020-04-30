import numpy as np 

class Quaternion:
    def __init__(self, q):
        if isinstance(q, np.ndarray):
            if q.shape == (4,) or q.shape == (4,1) or q.shape == (1,4):
                self.arr = q.squeeze()
            else:
                raise ValueError("Input must be a numpy array of length 4")
        else:
            raise ValueError("Input must be a numpy array of length 4")
    
    def __mul__(self, q):
        if isinstance(q, Quaternion):
            q_res = self.quatMul(q)

            if q_res.w < 0.0:
                q_res.arr *= -1
            return q_res
        else:
            raise ValueError("Input must be an instance of Quaternion")
    
    def quatMul(self, q):
            Q = self.w * np.eye(4) + np.block([[0, -self.v], [self.v[:, None], Quaternion.skew(self.v)]])
            q_res = Q @ q.arr 
            return Quaternion(q_res)
    
    def rot(self, v): #page 14 for a faster way possibly
        # q_v = Quaternion(np.hstack((0, v)))
        # q_mid = self.quatMul(q_v)
        # vp = q_mid.quatMul(self.inv())
        # return vp.arr[1:]
        q0 = self.w 
        qv = self.v 
        t = 2 * Quaternion.skew(v) @ qv
        vp = v + q0*t + Quaternion.skew(t) @ qv
        return vp
    
    def inv(self):
        q = self.arr.copy()
        q[1:] *= -1 
        return Quaternion(q)
    
    @staticmethod
    def skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    
    @property 
    def w(self):
        return self.arr[0]
    
    @property
    def v(self):
        return self.arr[1:]
    
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
            q = np.array([1/s * (R[2,0] - R[0,2]), 1/s * (R[1,0] + R[0,1]), s/4, 1/s * (R[2,1] + R[1,2])]) #Should the first be 1/2 or 1/s?
        else:
            s = 2 * np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
            q = np.array([1/s * (R[0,1] - R[1,0]), 1/s * (R[2,0] + R[0,2]), 1/s * (R[2,1] + R[1,2]), s/4])

        if q[0] < 0: #Always want the real part to be positive
            q *= -1
        
        return cls(q)
    
    @classmethod
    def fromAxisAngle(cls, vec):
        if not vec.shape == (3,) and not vec.shape == (3,1) and not vec.shape == (1,3):
            raise ValueError("Input must be a numpy array of length 3")

        theta = np.linalg.norm(vec)
        v = (vec / theta).squeeze() #Make a unit vector
        st2 = np.sin(theta/2)
        q = np.array([np.cos(theta/2), v[0] * st2, v[1] * st2, v[2] * st2])

        if q[0] < 0.0: #This shouldn't be necessary b/c theta should never be greater than pi rads
            q *= -1
        return cls(q)
    
    @classmethod 
    def fromRPY(cls, phi, theta, psi):
        c_phi = np.cos(phi/2)
        s_phi = np.sin(phi/2)
        c_theta = np.cos(theta/2)
        s_theta = np.sin(theta/2)
        c_psi = np.cos(psi/2)
        s_psi = np.sin(psi/2)

        q = np.zeros(4)
        q[0] = c_psi * c_theta * c_phi + s_psi * s_theta * s_phi  
        q[1] = c_psi * c_theta * s_phi - s_psi * s_theta * c_phi  
        q[2] = c_psi * s_theta * c_phi + s_psi * c_theta * s_phi  
        q[3] = s_psi * c_theta * c_phi - c_psi * s_theta * s_phi  

        if q[0] < 0.0:
            q *= -1
        return cls(q)
    
    @staticmethod 
    def log(q): #Taylor series expansion
        q0 = q.w
        qv = q.v
        theta = np.linalg.norm(qv)

        if np.abs(theta) > 1e-3:
            w = 2 * np.arctan(theta/q0) * qv/theta
        else:
            w = 2 * qv * (1/q0 - (theta**2)/(3 * q0**3) + (theta**4)/(5 * q0**5))
        return  np.hstack((0, w))
    
    @staticmethod
    def Log(q):
        logq = Quaternion.log(q)
        return Quaternion.vee(logq)

    @classmethod
    def exp(cls, w): 
        theta = np.linalg.norm(w)
        
        if np.abs(theta) > 2e-3:
            q0 = np.cos(theta/2)
            qv = np.sin(theta/2) * w[1:] / theta
        else:
            q0 = 1 - (theta/2)**2 + (theta/2)**4/24
            qv = (0.5 - (theta**2)/48 + (theta**4)/3840) * w[1:]
        if q0 < 0.0:
            q0 *= -1
            qv *= -1
        return cls(np.hstack((q0, qv)))
    
    @classmethod
    def Exp(cls, w):
        logq = cls.hat(w)
        return cls.exp(logq)
    
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
    
    @property
    def Adj(self):
        return self
    
    @property 
    def MatAdj(self):
        q0 = self.w 
        qv = self.v 
        R = (2*q0 - 1) * np.eye(3) - 2*q0*Quaternion.skew(qv) + 2 * np.outer(qv, qv)
        return R.T