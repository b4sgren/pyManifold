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

            p_true = q * Quaternion.Exp(w)
            p = Quaternion.Exp(q.Adj @ w) * q

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

    def test_right_jacobian_of_inversion(self):
        q = Quaternion.random()
        q_inv, Jr = q.inv(Jr=np.eye(3))
        Jr_true = -q.R

        np.testing.assert_allclose(Jr_true, Jr)

    def test_left_jacobian_of_inversion(self):
        q = Quaternion.random()
        q_inv, Jr = q.inv(Jr=np.eye(3))
        _, Jl = q.inv(Jl=np.eye(3))

        Adj_q = q.Adj
        Adj_qinv = q_inv.Adj
        Jl_true = Adj_qinv @ Jr @ np.linalg.inv(Adj_q)

        np.testing.assert_allclose(Jl_true, Jl)

    def test_right_jacobian_of_composition(self):
        q1 = Quaternion.random()
        q2 = Quaternion.random()

        q3, Jr = q1.compose(q2, Jr=np.eye(3))
        Jr_true = np.linalg.inv(q2.Adj)

        np.testing.assert_allclose(Jr_true, Jr)

    def test_left_jacobian_of_composition(self):
        for i in range(100):
            q1 = Quaternion.random()
            q2 = Quaternion.random()

            q3, Jr = q1.compose(q2, Jr=np.eye(3))
            _, Jl = q1. compose(q2, Jl=np.eye(3))

            Jl_true = q3.Adj @ Jr @ q1.inv().Adj

            np.testing.assert_allclose(Jl_true, Jl, atol=1e-10)

    def test_jacobians_of_exponential(self):
        for i in range(100):
            tau = np.random.uniform(-np.pi, np.pi, size=3)
            q, Jr = Quaternion.Exp(tau, Jr=np.eye(3))
            _, Jl = Quaternion.Exp(-tau, Jl=np.eye(3))

            np.testing.assert_allclose(Jl, Jr)

    def test_right_jacobian_of_logarithm(self):
        for i in range(100):
            q = Quaternion.random()
            logq, Jr_inv = Quaternion.Log(q, Jr=np.eye(3))
            _, Jr = Quaternion.Exp(logq, Jr=np.eye(3))

            np.testing.assert_allclose(np.linalg.inv(Jr), Jr_inv)

    def test_left_jacobian_or_logarithm(self):
        for i in range(100):
            q = Quaternion.random()
            logq, Jl_inv = Quaternion.Log(q, Jl=np.eye(3))
            _, Jl = Quaternion.Exp(logq, Jl=np.eye(3))

            np.testing.assert_allclose(np.linalg.inv(Jl), Jl_inv)

    def test_right_jacobian_of_rotation(self):
        for i in range(100):
            q = Quaternion.random()
            v = np.random.uniform(-10, 10, size=3)

            vp, Jr = q.rota(v, Jr=np.eye(3))
            vx = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
            Jr_true = - q.R @ vx

            np.testing.assert_allclose(Jr_true, Jr)

    def test_left_jacobian_of_rotation(self):
        for i in range(100):
            q = Quaternion.random()
            v = np.random.uniform(-10, 10, size=3)

            vp, Jl = q.rota(v, Jl=np.eye(3))
            _, Jr = q.rota(v, Jr=np.eye(3))

            Jl_true = np.eye(3) @ Jr @ q.Adj.T

            np.testing.assert_allclose(Jl_true, Jl, atol=1e-10)

    def test_jacobians_of_composition_second_element(self):
        for i in range(100):
            q1 = Quaternion.random()
            q2 = Quaternion.random()

            q3, Jr2 = q1.compose(q2, Jr2=np.eye(3))
            _, Jl2 = q1.compose(q2, Jl2=np.eye(3))

            Jl2_true = q3.Adj @ Jr2 @ np.linalg.inv(q2.Adj)

            np.testing.assert_allclose(Jl2_true, Jl2)

    '''
    These next two tests pass but the jacobians are the reverse of what James lists in his table. Sola makes no mentions of the jacobians being different so I'm not entirely sure what is right. I may need to check out Quaternions for the Error State Kalman filter to decide
    '''
    def test_right_jacobian_of_rotp(self):
        for i in range(100):
            q = Quaternion.random()
            v = np.random.uniform(-10, 10, size=3)

            vp, Jr = q.rotp(v, Jr=np.eye(3))
            # vx = np.array([[0, -v[2], v[1]],
                            # [v[2], 0, -v[0]],
                            # [-v[1], v[0], 0]])
            # Jr_true = q.R.T @ vx
            Jr_true = np.array([[0, -vp[2], vp[1]],
                                [vp[2], 0, -vp[0]],
                                [-vp[1], vp[0], 0]])

            np.testing.assert_allclose(Jr_true, Jr, atol=1e-10)

    def test_left_jacobian_of_rotp(self):
        for i in range(100):
            q = Quaternion.random()
            v = np.random.uniform(-10, 10, size=3)

            vp, Jl = q.rotp(v, Jl=np.eye(3))
            # Jl_true = np.array([[0, -vp[2], vp[1]],
                                # [vp[2], 0, -vp[0]],
                                # [-vp[1], vp[0], 0]])
            vx = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
            Jl_true = q.R.T @ vx

            np.testing.assert_allclose(Jl_true, Jl)

    def test_jacobians_of_boxplusr(self):
        for i in range(100):
            q = Quaternion.random()
            theta = np.random.uniform(-np.pi, np.pi, size=3)

            q2, Jr = q.boxplusr(theta, Jr=np.eye(3))
            _, Jl = q.boxplusr(theta, Jl=np.eye(3))

            Jl_true = q2.Adj @ Jr @ np.eye(3)

            np.testing.assert_allclose(Jl_true, Jl)

    def test_right_jacobians_of_boxminusr(self):
        for i in range(100):
            q1, q2 = Quaternion.random(), Quaternion.random()

            theta, Jr1 = q1.boxminusr(q2, Jr1=np.eye(3))
            dq = q2.inv() * q1
            _, Jr1_true = Quaternion.Log(dq, Jr=np.eye(3))

            _, Jr2 = q1.boxminusr(q2, Jr2=np.eye(3))
            _, Jr2_true = Quaternion.Log(dq, Jl=np.eye(3))

            np.testing.assert_allclose(Jr1_true, Jr1)
            np.testing.assert_allclose(-Jr2_true, Jr2)

    def test_left_jacobians_of_boxminusr(self):
        for i in range(100):
            q1, q2 = Quaternion.random(), Quaternion.random()

            theta, Jl1 = q1.boxminusr(q2, Jl1=np.eye(3))
            _, Jr1 = q1.boxminusr(q2, Jr1=np.eye(3))
            Jl1_true = np.eye(3) @ Jr1 @ q1.Adj.T

            _, Jl2 = q1.boxminusr(q2, Jl2=np.eye(3))
            _, Jr2 = q1.boxminusr(q2, Jr2=np.eye(3))
            Jl2_true = np.eye(3) @ Jr2 @ q2.Adj.T

            np.testing.assert_allclose(Jl1_true, Jl1)
            np.testing.assert_allclose(Jl2_true, Jl2)

    def test_jacobians_of_boxplusl(self):
        for i in range(100):
            q1 = Quaternion.random()
            theta = np.random.uniform(-np.pi, np.pi, size=3)

            q2, Jr = q1.boxplusl(theta, Jr=np.eye(3))
            _, Jl = q1.boxplusl(theta, Jl=np.eye(3))

            Jl_true = q2.Adj @ Jr @ np.eye(3)
            np.testing.assert_allclose(Jl_true, Jl)

    def test_jacobians_of_boxminusl(self):
        for i in range(100):
            q1, q2 = Quaternion.random(), Quaternion.random()

            theta, Jr = q1.boxminusl(q2, Jr1=np.eye(3))
            _, Jl = q1.boxminusl(q2, Jl1=np.eye(3))

            Jl_true = np.eye(3) @ Jr @ q1.Adj.T
            np.testing.assert_allclose(Jl_true, Jl)

    def test_jacobian_of_boxminusl_second_element(self):
        for i in range(100):
            q1, q2 = Quaternion.random(), Quaternion.random()

            theta, Jr2 = q1.boxminusl(q2, Jr2=np.eye(3))
            _, Jl2 = q1.boxminusl(q2, Jl2=np.eye(3))

            Jl_true = np.eye(3) @ Jr2 @ q2.Adj.T
            np.testing.assert_allclose(Jl_true, Jl2)


if __name__=="__main__":
    unittest.main()
