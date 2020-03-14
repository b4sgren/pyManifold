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

            np.testing.assert_allclose(T_true, T.arr) 
    
    def testLog(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=3)
            R = Rotation.random().as_dcm()

            T = SE3(t, R)
            logT = SE3.log(T)
            logT_true = sp.linalg.logm(T.arr)

            np.testing.assert_allclose(logT_true, logT, atol=1e-8)
        
        for i in range(100): #Test taylor series expansion
            t = np.random.uniform(-10, 10, size=3)
            ang = np.random.uniform(-1e-3, 1e-3)
            vec = np.random.uniform(-1.0, 1.0, size=3)
            vec = vec / np.linalg.norm(vec) * ang

            R = Rotation.from_rotvec(vec).as_dcm()
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

            np.testing.assert_allclose(T_true, T.arr)
        
        for i in range(100): #Test small thetas
            u = np.random.uniform(-10, 10, size=3)
            w = np.random.uniform(-1.0, 1.0, size=3)
            ang = np.random.uniform(-1e-3, 1e-3)
            w = w / np.linalg.norm(w) * ang

            arr = np.concatenate((w, u))

            T_true = sp.linalg.expm(SE3.hat(arr))
            T = SE3.Exp(arr)

            np.testing.assert_allclose(T_true, T.arr)
    
    def testVee(self):
        for i in range(100):
            u = np.random.uniform(-10, 10, size=3)
            w = np.random.uniform(-np.pi, np.pi, size=3)

            arr_true = np.hstack((w,u))

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

            u = np.random.uniform(-1.0, 1.0, size=3)
            w = np.random.uniform(-np.pi, np.pi, size=3)
            delta = np.concatenate((w, u))

            Adj_T = T.Adj

            T1_true = T * SE3.Exp(delta)
            T1 = SE3.Exp(Adj_T @ delta) * T

            np.testing.assert_allclose(T1_true.arr, T1.arr) #This one is not working
    
    def testInv(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=3)
            R = Rotation.random().as_dcm()

            T = SE3(t, R)

            T_inv = T.inv()
            T_inv_true = np.linalg.inv(T.arr)

            np.testing.assert_allclose(T_inv_true, T_inv.arr)
    
    def testGroupAction(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=3)
            t2 = np.random.uniform(-10, 10, size=3)
            R = Rotation.random().as_dcm()
            R2 = Rotation.random().as_dcm()

            T1 = SE3(t, R)
            T2 = SE3(t2, R2)

            T3 = T1 * T2

            R3 = R @ R2
            t3 = R @ t2 + t
            T3_true = SE3(t3, R3)

            np.testing.assert_allclose(T3_true.arr, T3.arr)
        
        for i in range(100):
            t = np.random.uniform(-10, 10, size=3)
            R = Rotation.random().as_dcm()

            T = SE3(t, R)

            pt = np.random.uniform(-5, 5, size=3)

            rot_pt = T * pt 
            rot_pt_true = T.R @ pt + T.t

            np.testing.assert_allclose(rot_pt_true, rot_pt)
    
    def testFromRPY(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=3)
            rpy = np.random.uniform(-np.pi, np.pi, size=3)

            T = SE3.fromRPY(t, rpy)
            R_true = Rotation.from_euler('ZYX', [rpy[2], rpy[1], rpy[0]]).as_dcm()
            T_true = SE3(t, R_true)

            np.testing.assert_allclose(T_true.arr, T.arr)
    
    def testFromAxisAngle(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=3)
            ang = np.random.uniform(-np.pi, np.pi)
            vec = np.random.uniform(-1, 1, size=3)
            vec = vec / np.linalg.norm(vec) * ang 

            T = SE3.fromAxisAngle(t, vec)
            R_true = Rotation.from_rotvec(vec).as_dcm()
            T_true = SE3(t, R_true)

            np.testing.assert_allclose(T_true.arr, T.arr)

if __name__=="__main__":
    unittest.main()