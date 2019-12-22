import unittest
import sys 
sys.path.append('..')
from so2 import SO2
import numpy as np

from IPython.core.debugger import Pdb

class SO2Test(unittest.TestCase):

    # def testExp(self):
    #     theta = np.pi/2.0
    #     R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    #     theta_skew = 

    def testLog(self):
        for i in range(100):
            theta = np.random.uniform(-np.pi, np.pi)
            R = SO2(theta)
            logR_true = np.array([[0, -theta], [theta, 0]])
            logR = SO2.log(R)
            self.assertAlmostEqual(logR[1,0], logR_true[1,0])
            self.assertAlmostEqual(logR[0,1], logR_true[0,1])

    # def testHat(self):
    #     debug = 1

    # def testVee(self):
    #     debug = 1

    # def testBoxPlus(self):
    #     debug = 1

    # def testBoxMinus(self):
    #     debug = 1

if __name__=="__main__":
    unittest.main()