import unittest
import sys 
sys.path.append('..')
from so2 import SO2
import numpy as np

from IPython.core.debugger import Pdb

class SO2Test(unittest.TestCase):

    def testExp(self):
        for i in range(100):
            theta = np.random.uniform(-np.pi, np.pi)
            logR = np.array([[0, -theta], [theta, 0]])
            R = SO2.exp(logR)
            R_true = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            self.assertAlmostEqual(R.arr[0,0], R_true[0,0])
            self.assertAlmostEqual(R.arr[1,0], R_true[1,0])
            self.assertAlmostEqual(R.arr[0,1], R_true[0,1])
            self.assertAlmostEqual(R.arr[1,1], R_true[1,1])

    def testLog(self):
        for i in range(100):
            theta = np.random.uniform(-np.pi, np.pi)
            R = SO2(theta)
            logR_true = np.array([[0, -theta], [theta, 0]])
            logR = SO2.log(R)
            self.assertAlmostEqual(logR[1,0], logR_true[1,0])
            self.assertAlmostEqual(logR[0,1], logR_true[0,1])

    def testHat(self):
        for i in range(100):
            theta = np.random.uniform(-np.pi, np.pi)
            logR_true = np.array([[0, -theta], [theta, 0]])
            logR = SO2.hat(theta)
            self.assertAlmostEqual(logR[0,1], logR_true[0,1])
            self.assertAlmostEqual(logR[1,0], logR_true[1,0])

    def testVee(self):
        for i in range(100):
            theta_true = np.random.uniform(-np.pi, np.pi)
            R = SO2(theta_true)
            theta = SO2.vee(SO2.log(R))
            self.assertAlmostEqual(theta, theta_true)
    
    def testMul(self):
        for i in range(100):
            theta1 = np.random.uniform(-np.pi, np.pi)
            theta2 = np.random.uniform(-np.pi, np.pi)
            R1 = SO2(theta1)
            R2 = SO2(theta2)
            R = R1 * R2
            theta_true = theta1 + theta2
            if theta_true > np.pi:
                theta_true -= 2 * np.pi 
            if theta_true < -np.pi:
                theta_true += 2 * np.pi
            R_true = np.array([[np.cos(theta_true), -np.sin(theta_true)], [np.sin(theta_true), np.cos(theta_true)]])
            self.assertAlmostEqual(R_true[0,0], R.arr[0,0])
            self.assertAlmostEqual(R_true[0,1], R.arr[0,1])
            self.assertAlmostEqual(R_true[1,0], R.arr[1,0])
            self.assertAlmostEqual(R_true[1,1], R.arr[1,1])

    # def testBoxPlus(self):
    #     debug = 1

    # def testBoxMinus(self):
    #     debug = 1

if __name__=="__main__":
    unittest.main()