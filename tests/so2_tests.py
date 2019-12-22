import unittest
import sys 
sys.path.append('..')
from so2 import SO2
import numpy as np

class SO2Test(unittest.TestCase):

    # def testExp(self):
    #     theta = np.pi/2.0
    #     R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    #     theta_skew = 

    def testLog(self):
        theta = np.pi/2.0
        R = SO2(theta)
        logR_true = np.array([[0, -theta], [theta, 0]])
        logR = SO2.log(R.arr)
        self.assertTrue((logR == logR_true).all())

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