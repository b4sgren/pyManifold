import numpy as np 
import scipy as sp
from scipy.spatial.transform import Rotation
import unittest
import sys
sys.path.append("..")
from so3 import SO3

class SO3_testing(unittest.TestCase):
    def testConstructor(self):
        for i in range(100):
            angles = np.random.uniform(-np.pi, np.pi, size=3)
            R_ex = SO3.fromRPY(angles)

            cp = np.cos(angles[0])
            sp = np.sin(angles[0])
            R1 = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])

            ct = np.cos(angles[1])
            st = np.sin(angles[1])
            R2 = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])

            cps = np.cos(angles[2])
            sps = np.sin(angles[2])
            R3 = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])

            R_true = Rotation.from_euler('ZYX', [angles[2], angles[1], angles[0]]).as_dcm() 

            R_ex2 = SO3(R_true)

            np.testing.assert_allclose(R_true, R_ex.arr)
            np.testing.assert_allclose(R_true, R_ex2.arr)
    
    def testLog(self):
        for i in range(100):
            temp = Rotation.random().as_dcm()
            R = SO3(temp)
            
            logR = SO3.log(R)

            logR_true = sp.linalg.logm(temp)

            np.testing.assert_allclose(logR_true, logR, atol=1e-10)
    
    def testExp(self):
        for i in range(100):
            logR_vec = np.random.uniform(-np.pi, np.pi, size=3)
            logR = np.array([[0, -logR_vec[2], logR_vec[1]],
                        [logR_vec[2], 0, -logR_vec[0]],
                        [-logR_vec[1], logR_vec[0], 0]])

            R = SO3.exp(logR)
            R_true = sp.linalg.expm(logR)

            np.testing.assert_allclose(R_true, R.arr)
    
    def testVee(self): #Is the result an axis-angle representation? aka a rotation vector?
        for i in range(100):
            omega_true = np.random.uniform(-np.pi, np.pi, size=3)
            logR = np.array([[0, -omega_true[2], omega_true[1]],
                            [omega_true[2], 0, -omega_true[0]],
                            [-omega_true[1], omega_true[0], 0]])

            omega = SO3.vee(logR)
            np.testing.assert_allclose(omega_true, omega)
    
    def testHat(self):
        for i in range(100):
            omega = np.random.uniform(-np.pi, np.pi, size=3)

            logR_true = np.array([[0, -omega[2], omega[1]],
                                [omega[2], 0, -omega[0]],
                                [-omega[1], omega[0], 0]])
            
            logR = SO3.hat(omega)

            np.testing.assert_allclose(logR_true, logR)
    
    def testInv(self):
        for i in range(100):
            mat = Rotation.random().as_dcm()
            R = SO3(mat)

            R_inv = R.inv()
            R_inv_true = np.linalg.inv(mat)

            np.testing.assert_allclose(R_inv_true, R_inv.arr)
    
    def testFromAxisEuler(self):
        for i in range(100):
            theta = np.random.uniform(-np.pi, np.pi)
            vec = np.random.uniform(-1, 1, size=3)
            vec = vec / np.linalg.norm(vec)

            R = SO3.fromAxisAngle(theta * vec)
            R_true = Rotation.from_rotvec(vec*theta).as_dcm()

            np.testing.assert_allclose(R_true, R.arr)

if __name__=="__main__":
    unittest.main()