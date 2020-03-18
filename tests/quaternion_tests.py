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

    # def testHat(self):
        # debug = 1

if __name__=="__main__":
    unittest.main()