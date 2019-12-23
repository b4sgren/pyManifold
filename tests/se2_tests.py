import unittest
import sys 
sys.path.append("..")
import numpy as np 
from se2 import SE2 

class SE2_Test(unittest.TestCase):
    def testInv(self):
        t = np.array([1.0, 0.0])
        theta = np.pi/2.0
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