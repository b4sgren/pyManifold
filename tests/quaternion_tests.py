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
    q[1:] = q[:3]
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
    
    def testFromRot(self): #This test fails. Need to determine if the quaternion used by Scipy is the same format as my quaternion
        for i in range(100):
            R = Rotation.random().as_dcm()
            q1 = Quaternion.fromRotationMatrix(R)
            q_true = Rotation.from_dcm(R).as_quat()
            q_true = switchOrder(q_true)

            if np.linalg.norm(q_true - q1.arr) > 1e-3:
                Pdb().set_trace()
                q2 = Quaternion.fromRotationMatrix(R)

            np.testing.assert_allclose(q_true, q1.arr)
    
    def testFromAxisAngle(self):
        for i in range(100):
            vec = Rotation.random().as_rotvec()
            q = Quaternion.fromAxisAngle(vec)
            q_true = Rotation.from_rotvec(vec).as_quat()
            q_true = switchOrder(q_true)

            if np.linalg.norm(q_true - q.arr) > 1e-3:
                Pdb().set_trace()
                q2 = Quaternion.fromAxisAngle(vec)

            np.testing.assert_allclose(q_true, q.arr)

if __name__=="__main__":
    unittest.main()