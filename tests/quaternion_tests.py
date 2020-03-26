import numpy as np 
import scipy as sp 
from scipy.spatial.transform import Rotation
import unittest 
import sys 
sys.path.append('..')
from quaternion import Quaternion

from IPython.core.debugger import Pdb 

def switchOrder(q):
    temp = q[-1]
    q[1:] = q[:3] # To get rotation matrices to work I need to negate this
    q[0] = temp 

    return q

class Quaternion_Testing(unittest.TestCase):
    def testVee(self):
        for i in range(100):
            q = Rotation.random().as_quat() #Note that the scalar is the last element
            q = switchOrder(q)

            w_true = q[1:]
            w = Quaternion.vee(q)

            np.testing.assert_allclose(w_true, w)

    def testHat(self):
        for i in range(100):
            vec = np.random.uniform(-1.0, 1.0, size=3)

            log_q = Quaternion.hat(vec)
            log_q_true = np.array([0, vec[0], vec[1], vec[2]])

            np.testing.assert_allclose(log_q_true, log_q)
    
    def testFromRot(self): #This test fails. All values are correct but sometimes the vector part has the wrong sign
        for i in range(100):
            R = Rotation.random().as_dcm()
            q1 = Quaternion.fromRotationMatrix(R)
            theta = np.arccos((np.trace(R) - 1)/2) 
            v = -np.array([R[2,1]-R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / (2 * np.sin(theta))
            q_true = np.zeros(4)
            q_true[0] = np.cos(theta/2)
            q_true[1:] = np.sin(theta/2) * v
 
            if np.sign(q_true[0]) < 0.0:
                q_true *= -1
 
            if np.linalg.norm(q_true - q1.arr) > 1e-3:
                Pdb().set_trace()
                q2 = Quaternion.fromRotationMatrix(R)
 
            np.testing.assert_allclose(q_true, q1.arr)
    
    def testFromAxisAngle(self):
        for i in range(100):
            vec = Rotation.random().as_rotvec()
            q1 = Quaternion.fromAxisAngle(vec)
            q_true = Rotation.from_rotvec(vec).as_quat()
            q_true = switchOrder(q_true)

            if np.linalg.norm(q_true - q1.arr) > 1e-3:
                Pdb().set_trace()
                q2 = Quaternion.fromAxisAngle(vec)

            np.testing.assert_allclose(q_true, q1.arr)
    

if __name__=="__main__":
    unittest.main()