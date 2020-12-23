import numpy as np
import scipy as sp
import scipy.linalg as spl
import unittest
import sys
sys.path.append('..')
from quaternion import Quaternion
from so3 import SO3

class Quaternion_Testing(unittest.TestCase):
    def testRandomGeneration(self):
        for i in range(100):
            q = Quaternion.random()
            q_norm = np.linalg.norm(q.q)

            np.testing.assert_allclose(1.0, q_norm)

    def testR(self):
        for i in range(100):
            q = Quaternion.random()
            R = SO3.fromQuaternion(q.q)

            np.testing.assert_allclose(q.R, R.R)

    def testQuaternionMultiply(self):
        for i in range(100):
            q1 = Quaternion.random()
            q2 = Quaternion.random()

            q3 = q1 * q2

            q3_true = np.array([q1.qw * q2.qw - q1.qx * q2.qx - q1.qy * q2.qy - q1.qz * q2.qz,
                                q1.qw * q2.qx + q1.qx * q2.qw + q1.qy * q2.qz - q1.qz * q2.qy,
                                q1.qw * q2.qy - q1.qx * q2.qz + q1.qy * q2.qw + q1.qz * q2.qx,
                                q1.qw * q2.qz + q1.qx * q2.qy - q1.qy * q2.qx + q1.qz * q2.qw])

            if q3_true[0] < 0:
                q3_true *= -1

            np.testing.assert_allclose(q3_true, q3.q)

    def testInverse(self):
        for i in range(100):
            q = Quaternion.random()
            q_inv = q.inv()

            I = q * q_inv
            I_true = np.array([1.0, 0., 0., 0.])

            np.testing.assert_allclose(I_true, I.q)

    def testRotationMatrixFromQuaternion(self):
        for i in range(100):
            # R_true = SO3.random().R
            # q = Quaternion.fromRotationMatrix(R_true)
            # R = q.R

            # np.testing.assert_allclose(R_true, R)
            q = Quaternion.random()
            R = q.R
            q2 = Quaternion.fromRotationMatrix(R)

            np.testing.assert_allclose(q.R, q2.R)

    def testRotatingVector(self):
        for i in range(100):
            v = np.random.uniform(-10, 10, size=3)
            q = Quaternion.random()
            R = q.R

            vp_true = R @ v
            vp = q.rota(v)

            np.testing.assert_allclose(vp_true, vp)

        for i in range(100):
            v = np.random.uniform(-10, 10, size=3)
            q = Quaternion.random()
            R = q.R

            vp_true = R.T @ v
            vp = q.rotp(v)

            np.testing.assert_allclose(vp_true, vp)

    def testFromRPY(self):
        for i in range(100):
            rpy = np.random.uniform(-np.pi, np.pi, size=3)
            R = SO3.fromRPY(rpy).R
            q = Quaternion.fromRPY(rpy)
            q_true = Quaternion.fromRotationMatrix(R)

            np.testing.assert_allclose(q_true.q, q.q)

    def testFromAxisAngle(self):
        for i in range(100):
            theta = np.random.uniform(0, np.pi)
            v = np.random.uniform(-10, 10, size=3)
            vec = theta * v/np.linalg.norm(v)

            R = SO3.fromAxisAngle(vec).R
            q = Quaternion.fromAxisAngle(vec)
            q_true = Quaternion.fromRotationMatrix(R)

            np.testing.assert_allclose(q_true.q, q.q)

    def testFromAxisAngleTaylor(self):
        for i in range(100): #Taylor series
            theta = np.random.uniform(0, 1e-3)
            v = np.random.uniform(-10, 10, size=3)
            vec = theta * v / np.linalg.norm(v)

            R = SO3.fromAxisAngle(vec).R
            q = Quaternion.fromAxisAngle(vec)
            q_true = Quaternion.fromRotationMatrix(R)

            np.testing.assert_allclose(q_true.q, q.q)

    def testHat(self):
        for i in range(100):
            w = np.random.uniform(-10, 10, size=3)

            W = Quaternion.hat(w)
            W_true = np.array([0, w[0], w[1], w[2]])

            np.testing.assert_allclose(W_true, W)

    def testVee(self):
        for i in range(100):
            W = np.random.uniform(-10, 10, size=4)
            W[0] = 0

            w_true = W[1:]
            w = Quaternion.vee(W)

            np.testing.assert_allclose(w_true, w)

    def testLog(self):
        for i in range(100):
            q = Quaternion.random()
            R = SO3.fromQuaternion(q.q)

            w_true = SO3.Log(R)
            w = Quaternion.Log(q)

            np.testing.assert_allclose(w_true, w)

    def testLogTaylor(self):
        for i in range(100):
            theta = np.random.uniform(0, 1e-3)
            v = np.random.uniform(-10, 10, size=3)
            vec = theta * v / np.linalg.norm(v)

            q = Quaternion.fromAxisAngle(vec)
            w = Quaternion.Log(q)

            np.testing.assert_allclose(vec, w)

    def testExp(self):
        for i in range(100):
            theta = np.random.uniform(-np.pi, np.pi)
            v = np.random.uniform(-1.0, 1.0, size=3)
            w = theta * v / np.linalg.norm(v)

            R = SO3.Exp(w)
            q_true = Quaternion.fromRotationMatrix(R.R)
            q = Quaternion.Exp(w)

            np.testing.assert_allclose(q_true.q, q.q)

    def testAdj(self):
        for i in range(100):
            q = Quaternion.random()
            w = np.random.uniform(-np.pi, np.pi, size=3)

            #Because of flipped order
            p_true = Quaternion.Exp(w) * q
            p = q * Quaternion.Exp(q.Adj @ w)

            np.testing.assert_allclose(p_true.q, p.q)

    def testNorm(self):
        for i in range(10):
            q = Quaternion.random()
            for i in range(10):
                q = q * q
            q.normalize()

            np.testing.assert_allclose(1, q.norm())

    def testBoxPlusR(self):
        for i in range(100):
            q = Quaternion.random()
            R = SO3.fromQuaternion(q.q)
            w = np.random.uniform(-1., 1., size=3)

            q2 = q.boxplusr(w)
            R2 = R.boxplusr(w)

            np.testing.assert_allclose(R2.R, q2.R)

    def testBoxMinus(self):
        for i in range(100):
            q1 = Quaternion.random()
            q2 = Quaternion.random()
            R1 = SO3.fromQuaternion(q1.q)
            R2 = SO3.fromQuaternion(q2.q)

            w1 = q1.boxminusr(q2)
            w2 = R1.boxminusr(R2)

            np.testing.assert_allclose(w1, w2)

    def test_boxplusl(self):
        for i in range(100):
            q = Quaternion.random()
            R = SO3.fromQuaternion(q.q)
            w = np.random.uniform(-np.pi, np.pi, size=3)

            q2 = q.boxplusl(w)
            R2 = R.boxplusl(w)

            np.testing.assert_allclose(R2.R, q2.R)

    def test_boxminusl(self):
        for i in range(100):
            q1 = Quaternion.random()
            q2 = Quaternion.random()

            w = q1.boxminusl(q2)
            q = q2.boxplusl(w)

            np.testing.assert_allclose(q1.q, q.q)

if __name__=="__main__":
    unittest.main()
