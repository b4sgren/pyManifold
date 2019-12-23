import unittest
import sys 
sys.path.append("..")
import numpy as np 
from se2 import SE2 

class SE2_Test(unittest.TestCase):
    def testInv(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)
            ct = np.cos(theta)
            st = np.sin(theta)
            R = np.array([[ct, -st], [st, ct]])

            T = SE2(R, t)
            T_inv = T.inv()
            T_inv_true = np.eye(3)
            T_inv_true[:2, :2] = R.T 
            T_inv_true[:2, 2] = -R.T @ t 

            np.testing.assert_allclose(T_inv_true, T_inv.arr)
    
    def testGroupOperator(self):
        for i in range(100):
            t1 = np.random.uniform(-10, 10, size=2)
            theta1 = np.random.uniform(-np.pi, np.pi)
            t2 = np.random.uniform(-10, 10, size=2)
            theta2 = np.random.uniform(-np.pi, np.pi)

            ct1 = np.cos(theta1)
            ct2 = np.cos(theta2)
            st1 = np.sin(theta1)
            st2 = np.sin(theta2)
            R1 = np.array([[ct1, -st1], [st1, ct1]])
            R2 = np.array([[ct2, -st2], [st2, ct2]])
            
            T1 = SE2(R1, t1)
            T2 = SE2(R2, t2)
            T = T1 * T2
            
            R_true = R1 @ R2
            t_true = R1 @ t2 + t1
            T_true = SE2(R_true, t_true)

            np.testing.assert_allclose(T_true.arr, T.arr)

if __name__=="__main__":
    unittest.main()