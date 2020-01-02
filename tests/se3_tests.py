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
    
    def testLog(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=3)
            R = Rotation.random().as_dcm()

            T = SE3(t, R)
            logT = SE3.log(T)

            logT_true = sp.linalg.logm(T.arr)

            np.testing.assert_allclose(logT_true, logT, atol=1e-8)
    
    def testExp(self):
        for i in range(100):
            u = np.random.uniform(-10, 10, size=3)
            w = np.random.uniform(-np.pi, np.pi, size=3)

            logR = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            
            logT = np.zeros((4,4))
            logT[:3,:3] = logR 
            logT[:3,3] = u 

            T_true = sp.linalg.expm(logT)
            T = SE3.exp(logT)

            np.testing.assert_allclose(T_true, T.arr, atol=1e-8)
    
    def testVee(self):
        for i in range(100):
            u = np.random.uniform(-10, 10, size=3)
            w = np.random.uniform(-np.pi, np.pi, size=3)

            arr_true = np.hstack((u,w))

            logR = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            logT = np.zeros((4,4))
            logT[:3,:3] = logR 
            logT[:3,3] = u
            
            arr = SE3.vee(logT)

            np.testing.assert_allclose(arr_true, arr)
    
    def testHat(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=3)
            R = Rotation.random().as_dcm()

            T = SE3(t, R)
            logT_true = SE3.log(T)
            arr = SE3.vee(logT_true)

            logT = SE3.hat(arr)

            np.testing.assert_allclose(logT_true, logT)
    
    def testAdj(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=3)
            R = Rotation.random().as_dcm()

            T = SE3(t, R)
            Adj = T.Adj

            u = np.random.uniform(-1, 1, size=3)
            w = np.random.uniform(-np.pi, np.pi, size=3)
            delta = np.hstack((u,w))

            delta2 = Adj @ delta
            delta2_true = np.zeros(6)
            delta2_true[-3:] = R @ w 
            delta2_true[:3] = R @ u + np.cross(t, R @ w)

            np.testing.assert_allclose(delta2_true, delta2)

if __name__=="__main__":
    unittest.main()