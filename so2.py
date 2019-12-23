import numpy as np 

G = np.array([[0, -1], [1, 0]])
class SO2:
    def __init__(self, theta):
        if isinstance(theta, float):
            self.arr = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]]) #Should the - sign be switched for our application? 
        elif isinstance(theta, np.ndarray):
            self.arr = theta
            
    def __mul__(self, R2):
        arr = self.arr @ R2.arr
        return SO2(arr)
   
    @classmethod
    def exp(cls, theta_x):
        theta = theta_x[1,0]
        return cls(theta)
    
    @staticmethod
    def log(R): 
        theta = np.arctan2(R.arr[1,0], R.arr[0,0])
        return G * theta

    @staticmethod
    def vee(theta_x):
        return theta_x[1,0]

    @staticmethod
    def hat(theta):
        return theta * G
    
    @classmethod
    def boxplus(cls, R, theta): #Needs to be tested
        R2 = cls(theta)
        return R * R2

    def boxminus(self, R2): #Needs to be tested
        temp = self.arr @ R2.arr.T #R2 doesn't have a transpose function
        theta = np.arctan2(temp.arr[1,0], temp.arr[0,0])
        return theta