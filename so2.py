import numpy as np 

G = np.array([[0, -1], [1, 0]])

#TODO: add jacobians, overload +/- for box plus and box minus
class SO2:
    def __init__(self, R):
        assert R.shape == (2,2)
        self.arr = R
            
    def __mul__(self, R2):
        if isinstance(R2, SO2):
            arr = self.arr @ R2.arr
            return SO2(arr)
        # elif isinstance(R2, np.ndarray):
            # if not R2.size == 2:
                # raise ValueError("Array being multiplied needs to be a 1D array of length 2")
            # else:
                # return self.arr @ R2
        else:
            raise ValueError("Type not supported. Make sure R2 is an SO2 object of a numpy array")
    
    def inv(self):
        return SO2(self.arr.T)
    
    def rota(self, v):
        assert v.size == 2
        return self.arr @ v
    
    def rotp(self, v):
        assert v.size == 2
        return self.inv().arr @ v
    
    @property 
    def R(self):
        return self.arr 
    
    @classmethod 
    def fromAngle(cls, theta):
        ct = np.cos(theta)
        st = np.sin(theta)
        R = np.array([[ct, -st], [st, ct]])
        return cls(R)

    @classmethod
    def exp(cls, theta_x):
        assert theta_x.shape == (2,2)
        theta = theta_x[1,0]
        return cls.fromAngle(theta)
    
    @classmethod 
    def Exp(cls, theta):
        logR = cls.hat(theta)
        return cls.exp(logR)
    
    @staticmethod
    def log(R): 
        assert isinstance(R, SO2)
        theta = np.arctan2(R.arr[1,0], R.arr[0,0])
        return G * theta
    
    @classmethod
    def Log(cls, R):
        logR = cls.log(R)
        return cls.vee(logR)

    @staticmethod
    def vee(theta_x):
        assert theta_x.shape == (2,2)
        return theta_x[1,0]

    @staticmethod
    def hat(theta):
        return theta * G
    
    @property 
    def Adj(self):
        return np.eye(2)