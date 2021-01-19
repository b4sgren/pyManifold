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
            R = SO2.fromAngle(theta)
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
            R = SO2.fromAngle(theta_true)
            theta = SO2.vee(SO2.log(R))
            self.assertAlmostEqual(theta, theta_true)

    def testMul(self):
        for i in range(100):
            theta1 = np.random.uniform(-np.pi, np.pi)
            theta2 = np.random.uniform(-np.pi, np.pi)
            R1 = SO2.fromAngle(theta1)
            R2 = SO2.fromAngle(theta2)
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

    def testRotateVector(self):
        for i in range(100): #Active rotation
            theta = np.random.uniform(-np.pi, np.pi)
            R = SO2.fromAngle(theta)

            pt = np.random.uniform(-5, 5, size=2)

            rot_pt = R.rota(pt)

            x_true = np.cos(theta) * pt[0] - np.sin(theta) * pt[1]
            y_true = np.sin(theta) * pt[0] + np.cos(theta) * pt[1]

            rot_pt_true = np.array([x_true, y_true])

            np.testing.assert_allclose(rot_pt_true, rot_pt)

        for i in range(100): #Passive rotation
            theta = np.random.uniform(-np.pi, np.pi)
            R = SO2.fromAngle(theta)
            pt = np.random.uniform(-5, 5, size=2)

            rot_pt = R.rotp(pt)

            x_true = np.cos(theta) * pt[0] + np.sin(theta) * pt[1]
            y_true = -np.sin(theta) * pt[0] + np.cos(theta) * pt[1]

            rot_pt_true = np.array([x_true, y_true])

            np.testing.assert_allclose(rot_pt_true, rot_pt)

    def testInv(self):
        for i in range(100):
            theta = np.random.uniform(-np.pi, np.pi)
            R = SO2.fromAngle(theta)
            mat = R.arr

            R_inv_true = np.linalg.inv(mat)
            R_inv = R.inv()

            np.testing.assert_allclose(R_inv_true, R_inv.arr)

    def testAdjoint(self):
        for i in range(100):
            theta = np.random.uniform(-np.pi, np.pi)
            R = SO2.fromAngle(theta)
            delta = np.random.uniform(-np.pi, np.pi)

            Adj_R = R.Adj

            Rf = R * SO2.Exp(delta)
            Rf_true = SO2.Exp(Adj_R * delta) * R

            np.testing.assert_allclose(Rf_true.R, Rf.R)

    def testBoxPlusR(self):
        for i in range(100):
            R = SO2.random()
            w = np.random.uniform(-np.pi, np.pi)
            R2 = SO2.fromAngle(w)

            R3 = R.boxplusr(w)
            R3_true = R * R2

            np.testing.assert_allclose(R3_true.R, R3.R)

    def testBoxMinusR(self):
        R1 = SO2.random()
        R2 = SO2.random()

        w = R1.boxminusr(R2)
        Rres = R2.boxplusr(w)

        np.testing.assert_allclose(R1.R, Rres.R)

    def test_boxplusl(self):
        for i in range(100):
            R = SO2.random()
            w = np.random.uniform(-np.pi, np.pi)
            R2 = SO2.fromAngle(w)

            R3 = R.boxplusl(w)
            R3_true = R2 * R

            np.testing.assert_allclose(R3_true.R, R3.R)

    def test_boxminusl(self):
        for i in range(100):
            R1 = SO2.random()
            R2 = SO2.random()

            w = R1.boxminusl(R2)
            R = R2.boxplusl(w)

            np.testing.assert_allclose(R.R, R1.R)

    def test_right_jacobian_of_inversion(self):
        R = SO2.random()
        R_inv, Jr = R.inv(Jr=True)

        self.assertEqual(-1, Jr)

    def test_left_jacobian_of_inversion(self):
        R = SO2.random()
        R_inv, Jl = R.inv(Jl=True)
        _, Jr = R.inv(Jr=True)

        Adj_R = R.Adj
        Adj_Rinv = R_inv.Adj

        Jl_true = Adj_Rinv * Jr * (1.0 / Adj_R)

        self.assertEqual(-1, Jl)
        self.assertEqual(Jl_true, Jl)

    def test_right_jacobian_of_composition(self):
        R1 = SO2.random()
        R2 = SO2.random()

        R3, Jr1 = R1.compose(R2, Jr=True)
        Jr1_true = 1.0/R2.Adj

        np.testing.assert_allclose(Jr1_true, Jr1)

    def test_left_jacobian_of_composition(self):
        R1 = SO2.random()
        R2 = SO2.random()

        R3, Jr = R1.compose(R2, Jr=True)
        _, Jl = R1.compose(R2, Jl=True)

        Adj_R1 = R1.Adj
        Adj_R3 = R3.Adj
        Jl_true = Adj_R3 * Jr * 1.0/Adj_R1

        np.testing.assert_allclose(Jl_true, Jl)

    def test_right_jacobian_of_composition_second_element(self):
        R1 = SO2.random()
        R2 = SO2.random()

        R3, Jr2 = R1.compose(R2, Jr2=True)
        Jr2_true = 1.0

        np.testing.assert_allclose(Jr2_true, Jr2)

    def test_left_jacobian_of_composition_second_element(self):
        R1 = SO2.random()
        R2 = SO2.random()

        R3, Jl2 = R1.compose(R2, Jl2=True)
        _, Jr2 = R1.compose(R2, Jr2=True)
        Jl2_true = R3.Adj * Jr2 * 1.0 / R2.Adj

        np.testing.assert_allclose(Jl2_true, Jl2)

    def test_right_jacobian_of_exponential(self):
        tau = np.random.uniform(-np.pi, np.pi)
        R, Jr = SO2.Exp(tau, Jr=True)

        self.assertEqual(1.0, Jr)

    def test_left_jacobian_of_exponential(self):
        tau = np.random.uniform(-np.pi, np.pi)
        R, Jr = SO2.Exp(tau, Jr=True)
        _, Jl = SO2.Exp(tau, Jl=True)

        Ad_R = Jl * (1.0 / Jr)

        self.assertEqual(1.0, Jl)
        self.assertEqual(Ad_R, R.Adj)

    def test_right_jacobian_of_logarithm(self):
        R = SO2.random()
        logR, Jr_inv = SO2.Log(R, Jr=True)
        _, Jr = SO2.Exp(logR, Jr=True)

        self.assertEqual(1/Jr, Jr_inv)

    def test_left_jacobian_of_logarithm(self):
        R = SO2.random()
        logR, Jl_inv = SO2.Log(R, Jl=True)
        _, Jl = SO2.Exp(logR, Jl=True)

        self.assertEqual(1/Jl, Jl_inv)

    def test_right_jacobian_of_rota(self):
        for i in range(100):
            R = SO2.random()
            v = np.random.uniform(-10, 10, size=2)

            vp, Jr = R.rota(v, Jr=True)
            vx = np.array([-v[1], v[0]])
            Jr_true = R.R @ vx

            np.testing.assert_allclose(Jr_true, Jr)

    def test_left_jacobian_of_rota(self):
        for i in range(100):
            R = SO2.random()
            v = np.random.uniform(-10, 10, size=2)

            vp, Jl = R.rota(v, Jl=True)
            _, Jr = R.rota(v, Jr=True)

            np.testing.assert_allclose(Jr, Jl)

    def test_right_jacobian_of_rotp(self):
        for i in range(100):
            R = SO2.random()
            v = np.random.uniform(-10, 10, size=2)

            vp, Jr = R.rotp(v, Jr=1)
            one_x = np.array([[0, -1], [1, 0]])
            Jr_true = -one_x @ vp

            np.testing.assert_allclose(Jr_true, Jr)

    def test_left_jacobian_of_rotp(self):
        for i in range(100):
            R = SO2.random()
            v = np.random.uniform(-10, 10, size=2)

            vp, Jl = R.rotp(v, Jl=1)
            one_x = np.array([[0, -1], [1, 0]])
            Jl_true = - R.R.T @ one_x @ v

            np.testing.assert_allclose(Jl_true, Jl)

    def test_right_jacobian_of_boxplusr(self):
        for i in range(100):
            R = SO2.random()
            theta = np.random.uniform(-np.pi, np.pi)

            R2, Jr = R.boxplusr(theta, Jr=1)
            _, Jr_true = SO2.Exp(theta, Jr=1)

            np.testing.assert_allclose(Jr_true, Jr)

    def test_left_jacobian_of_boxplusr(self):
        for i in range(100):
            R = SO2.random()
            theta = np.random.uniform(-np.pi, np.pi)

            R2, Jl = R.boxplusr(theta, Jl=1)
            _, Jr = R.boxplusr(theta, Jr=1)

            Jl_true = R2.Adj * Jr * 1.0 # Using adjoing formula

            np.testing.assert_allclose(Jl_true, Jl)

    def test_right_jacobians_or_boxminusr(self):
        for i in range(100):
            R1 = SO2.random()
            R2 = SO2.random()

            theta, Jr1 = R1.boxminusr(R2, Jr1=1)
            _, Jr2 = R1.boxminusr(R2, Jr2=1)
            dR = R2.inv() * R1
            _, Jr1_true = SO2.Log(dR, Jr=1)
            _, Jr2_true = SO2.Log(dR, Jl=1)

            np.testing.assert_allclose(Jr1_true, Jr1)
            np.testing.assert_allclose(Jr2_true, Jr2)

if __name__=="__main__":
    unittest.main()
