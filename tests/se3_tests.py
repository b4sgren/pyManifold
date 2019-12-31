import unittest
import numpy as np 
import scipy as sp 
from scipy.spatial.transform import Rotation
import sys 
sys.path.append('..')
from se3 import SE3

class SE3_Test(unittest.TestCase):
    def testConstructor(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=3)
            R = Rotation.random().as_dcm()

            T_true = np.eye(4)
            T_true[:3,:3] = R 
            T_true[:3,3] = t 

            T = SE3(t, R)

            np.testing.assert_allclose(T_true, T.arr)
        
        for i in range(100):
            t = np.random.uniform(-10, 10, size=3)
            angles = np.random.uniform(-np.pi, np.pi, size=3)
            R = Rotation.from_euler('ZYX', [angles[2], angles[1], angles[0]]).as_dcm()

            T = SE3(t, angles[0], angles[1], angles[2])

            T_true = np.eye(4)
            T_true[:3,:3] = R 
            T_true[:3,3] = t

            np.testing.assert_allclose(T_true, T.arr) #This test is failing. Not sure what is wrong

if __name__=="__main__":
    unittest.main()