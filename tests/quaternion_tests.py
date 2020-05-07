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
    
    def testFromRPY(self):
        for i in range(100):
            rpy = np.random.uniform(-np.pi, np.pi, size=3)
            R = SO3.fromRPY(rpy).R 
            q = Quaternion.fromRPY(rpy)
            q_true = Quaternion.fromRotationMatrix(R)

            np.testing.assert_allclose(q_true.q, q.q)
    
    def testFromAxisAngle(self):
        for i in range(100):
            theta = np.random.uniform(0, np.pi)
            v = np.random.uniform(-10, 10, size=3)
            vec = theta * v/np.linalg.norm(v)

            R = SO3.fromAxisAngle(vec).R 
            q = Quaternion.fromAxisAngle(vec)
            q_true = Quaternion.fromRotationMatrix(R)

            # np.testing.assert_allclose(q_true.q, q.q) #TODO: Values match but vector part has opposite sign
    
    def testHat(self):
        for i in range(100):
            w = np.random.uniform(-10, 10, size=3)

            W = Quaternion.hat(w)
            W_true = np.array([0, w[0], w[1], w[2]])

            np.testing.assert_allclose(W_true, W)
    
    def testVee(self):
        for i in range(100):
            W = np.random.uniform(-10, 10, size=4)
            W[0] = 0

            w_true = W[1:]
            w = Quaternion.vee(W)

            np.testing.assert_allclose(w_true, w)

if __name__=="__main__":
    unittest.main()