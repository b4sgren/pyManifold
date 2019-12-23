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

            # self.assertAlmostEqual()
            np.testing.assert_allclose(T_inv_true, T_inv.arr)

if __name__=="__main__":
    unittest.main()