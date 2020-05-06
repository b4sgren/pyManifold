import numpy as np 
import scipy as sp 
import unittest 
import sys 
sys.path.append('..')
from quaternion import Quaternion

from IPython.core.debugger import Pdb 

class Quaternion_Testing(unittest.TestCase):
    def testRandomGeneration(self):
        for i in range(100):
            q = Quaternion.random()
            q_norm = np.linalg.norm(q.q)

            np.testing.assert_allclose(1.0, q_norm)

if __name__=="__main__":
    unittest.main()