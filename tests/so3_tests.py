import numpy as np 
from scipy.spatial.transform import Rotation
import unittest
import sys
sys.path.append("..")
from so3 import SO3

class SO3_testing(unittest.TestCase):
    def testConstructor(self):
        for i in range(100):
            angles = np.random.uniform(-np.pi, np.pi, size=3)
            R_ex = SO3(angles[0], angles[1], angles[2])

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
    

if __name__=="__main__":
    unittest.main()