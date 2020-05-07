import numpy as np 
import scipy as sp 
import unittest 
import sys 
sys.path.append('..')
from quaternion import Quaternion
from so3 import SO3

from IPython.core.debugger import Pdb 

class Quaternion_Testing(unittest.TestCase):
    def testRandomGeneration(self):
        for i in range(100):
            q = Quaternion.random()
            q_norm = np.linalg.norm(q.q)

            np.testing.assert_allclose(1.0, q_norm)
        
    def testQuaternionMultiply(self):
        for i in range(100):
            q1 = Quaternion.random()
            q2 = Quaternion.random()

            q3 = q1 * q2

            q3_true = np.array([q1.qw * q2.qw - q1.qx * q2.qx - q1.qy * q2.qy - q1.qz * q2.qz,
                                q1.qw * q2.qx + q1.qx * q2.qw + q1.qy * q2.qz - q1.qz * q2.qy,
                                q1.qw * q2.qy - q1.qx * q2.qz + q1.qy * q2.qw + q1.qz * q2.qx,
                                q1.qw * q2.qz + q1.qx * q2.qy - q1.qy * q2.qx + q1.qz * q2.qw])
            
            if q3_true[0] < 0:
                q3_true *= -1
            
            np.testing.assert_allclose(q3_true, q3.q)
    
    def testInverse(self):
        for i in range(100):
            q = Quaternion.random()
            q_inv = q.inv()

            I = q * q_inv 
            I_true = np.array([1.0, 0., 0., 0.])

            np.testing.assert_allclose(I_true, I.q)
    
    def testRotationMatrixFromQuaternion(self):
        for i in range(100):
            R_true = SO3.random().R
            q = Quaternion.fromRotationMatrix(R_true)
            R = q.R

            np.testing.assert_allclose(R_true, R) #The diagonal is off
    
    def testRotatingVector(self):
        for i in range(100):
            v = np.random.uniform(-10, 10, size=3)
            q = Quaternion.random()
            R = q.R

            vp_true = R @ v
            vp = q.rot(v)

            np.testing.assert_allclose(vp_true, vp)

if __name__=="__main__":
    unittest.main()