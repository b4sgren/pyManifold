import unittest
import scipy.linalg as spl
import sys
sys.path.append("..")
import numpy as np
from se2 import SE2

from IPython.core.debugger import Pdb

class SE2_Test(unittest.TestCase):
    def testInv(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)
            ct = np.cos(theta)
            st = np.sin(theta)
            R = np.array([[ct, -st], [st, ct]])

            T = SE2.fromRandt(R, t)
            T_inv = T.inv()
            T_inv_true = np.eye(3)
            T_inv_true[:2, :2] = R.T
            T_inv_true[:2, 2] = -R.T @ t

            np.testing.assert_allclose(T_inv_true, T_inv.arr)

    def testGroupOperator(self):
        for i in range(100):
            t1 = np.random.uniform(-10, 10, size=2)
            theta1 = np.random.uniform(-np.pi, np.pi)
            t2 = np.random.uniform(-10, 10, size=2)
            theta2 = np.random.uniform(-np.pi, np.pi)

            ct1 = np.cos(theta1)
            ct2 = np.cos(theta2)
            st1 = np.sin(theta1)
            st2 = np.sin(theta2)
            R1 = np.array([[ct1, -st1], [st1, ct1]])
            R2 = np.array([[ct2, -st2], [st2, ct2]])

            T1 = SE2.fromRandt(R1, t1)
            T2 = SE2.fromRandt(R2, t2)
            T = T1 * T2

            R_true = R1 @ R2
            t_true = R1 @ t2 + t1
            T_true = SE2.fromRandt(R_true, t_true)

            np.testing.assert_allclose(T_true.arr, T.arr)

    def testActionOnVector(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)

            T = SE2.fromAngleAndt(theta, t)

            vec = np.random.uniform(-5, 5, size=2)

            pt = T.transa(vec)

            pt_true = T.R @ vec + T.t

            np.testing.assert_allclose(pt_true, pt)

        for i in range(100):
            T = SE2.random()
            vec = np.random.uniform(-5, 5, size=2)

            pt = T.transp(vec)
            pt_true = T.R.T @ vec - T.R.T @ T.t

            np.testing.assert_allclose(pt_true, pt)

    def testLog(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)

            ct = np.cos(theta)
            st = np.sin(theta)
            R = np.array([[ct, -st], [st, ct]])

            T = SE2.fromRandt(R, t)
            logT = SE2.log(T)
            logT_true = spl.logm(T.arr)
            if np.linalg.norm(logT_true - logT, ord='fro') > 1e-3:
                Pdb().set_trace()
                debug = 1
                temp = SE2.log(T)

            np.testing.assert_allclose(logT_true, logT, atol=1e-7)

    def testTaylorLog(self):
        for i in range(100): #Test taylor series
            t = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-1e-8, 1e-8)

            T = SE2.fromAngleAndt(theta, t)
            logT = SE2.log(T)
            logT_true = spl.logm(T.arr)

            if np.linalg.norm(logT_true - logT, ord='fro') > 1e-8:
                Pdb().set_trace()
                debug = 1
                temp = SE2.log(T)

            np.testing.assert_allclose(logT_true, logT, atol=1e-7)

    def testExp(self):
        for i in range(100):
            v = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)

            logT = np.array([[0, -theta, v[0]],
                            [theta, 0, v[1]],
                            [0, 0, 0]])

            T = SE2.exp(logT)

            T_true = spl.expm(logT)

            np.testing.assert_allclose(T_true, T.arr)

    def testTaylorExp(self):
        for i in range(100):
            v = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-1e-8, 1e-8)
            arr = np.array([v[0], v[1], theta])

            T = SE2.Exp(arr)
            T_true = spl.expm(SE2.hat(arr))

            np.testing.assert_allclose(T_true, T.arr)

    def testVee(self):
        for i in range(100):
            u = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)

            # arr_true = np.array([theta, u[0], u[1]])
            arr_true = np.array([u[0], u[1], theta])
            X = np.array([[0, -theta, u[0]], [theta, 0, u[1]], [0, 0, 0]])
            arr = SE2.vee(X)

            np.testing.assert_allclose(arr_true, arr)

    def testHat(self):
        for i in range(100):
            u = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)

            arr = np.array([u[0], u[1], theta])
            X_true = np.array([[0, -theta, u[0]], [theta, 0, u[1]], [0, 0, 0]])
            X = SE2.hat(arr)

            np.testing.assert_allclose(X_true, X)

    def testAdjoint(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)
            u = np.random.uniform(-1, 1, size=2)
            phi = np.random.uniform(-np.pi, np.pi)
            # delta = np.array([phi, u[0], u[1]])
            delta = np.array([u[0], u[1], phi])

            ct = np.cos(theta)
            st = np.sin(theta)
            R = np.array([[ct, -st], [st, ct]])
            T = SE2.fromRandt(R, t)

            adj = T.Adj

            T2_true = T * SE2.Exp(delta)
            T2 = SE2.Exp(adj @ delta) * T

            np.testing.assert_allclose(T2_true.arr, T2.arr)

    def testFromAngle(self):
        for i in range(100):
            t = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)

            T = SE2.fromAngleAndt(theta, t)

            ct = np.cos(theta)
            st = np.sin(theta)
            R = np.array([[ct, -st], [st, ct]])
            T_true = np.eye(3)
            T_true[:2,:2] = R
            T_true[:2,2] = t

            np.testing.assert_allclose(T_true, T.arr)

    def testBoxPlusR(self):
        for i in range(100):
            T = SE2.random()
            u = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)
            vec = np.array([*u, theta])

            T3 = T.boxplusr(vec)
            T3_true = T * SE2.Exp(vec)

            np.testing.assert_allclose(T3_true.T, T3.T)

    def testBoxMinusR(self):
        for i in range(100):
            T1 = SE2.random()
            T2 = SE2.random()

            w = T1.boxminusr(T2)
            T = T2.boxplusr(w)

            np.testing.assert_allclose(T1.T, T.T)

    def test_boxplusl(self):
        for i in range(100):
            T = SE2.random()
            u = np.random.uniform(-10., 10., size=2)
            theta = np.random.uniform(-np.pi, np.pi)
            vec = np.array([*u, theta])

            T2 = T.boxplusl(vec)
            T2_true = SE2.Exp(vec) * T

            np.testing.assert_allclose(T2_true.T, T2.T)

    def test_boxminusl(self):
        for i in range(100):
            T1 = SE2.random()
            T2 = SE2.random()

            w = T1.boxminusl(T2)
            T = T2.boxplusl(w)

            np.testing.assert_allclose(T.T, T1.T)

    def test_right_jacobian_of_inversion(self):
        T = SE2.random()
        T_inv, Jr = T.inv(Jr=np.eye(3))

        np.testing.assert_allclose(-T.Adj, Jr)

    def test_left_jacobian_of_inversion(self):
        T = SE2.random()
        T_inv, Jr = T.inv(Jr=np.eye(3))
        _, Jl = T.inv(Jl=np.eye(3))

        Adj_T = T.Adj
        Adj_Tinv = T_inv.Adj

        Jl_true = Adj_Tinv @ Jr @ np.linalg.inv(Adj_T)

        np.testing.assert_allclose(Jl_true, Jl)

    def test_right_jacobian_of_composition(self):
        T1 = SE2.random()
        T2 = SE2.random()

        T3, Jr = T1.compose(T2, Jr=np.eye(3))
        Jr_true = np.linalg.inv(T2.Adj)

        np.testing.assert_allclose(Jr_true, Jr)

    def test_left_jacobian_of_composition(self):
        for i in range(100):
            T1 = SE2.random()
            T2 = SE2.random()

            T3, Jr = T1.compose(T2, Jr=np.eye(3))
            _, Jl = T1.compose(T2, Jl=np.eye(3))

            Jl_true = T3.Adj @ Jr @ T1.inv().Adj

            np.testing.assert_allclose(Jl_true, Jl, atol=1e-10)

    def test_jacobians_of_exponential(self):
        t = np.random.uniform(-10.0, 10.0, size=2)
        theta = np.random.uniform(-np.pi, np.pi)
        tau = np.array([*t, theta])

        T, Jr = SE2.Exp(tau, Jr=np.eye(3))
        _, Jl = SE2.Exp(-tau, Jl=np.eye(3))

        np.testing.assert_allclose(Jr, Jl)

    def test_right_jacobian_of_logarithm(self):
        T = SE2.random()
        tau, Jr_inv = SE2.Log(T, Jr=np.eye(3))
        _, Jr = SE2.Exp(tau, Jr=np.eye(3))

        np.testing.assert_allclose(np.linalg.inv(Jr), Jr_inv)

    def test_left_jacobian_or_logarithm(self):
        T = SE2.random()
        tau, Jl_inv = SE2.Log(T, Jl=np.eye(3))
        _, Jl = SE2.Exp(tau, Jl=np.eye(3))

        np.testing.assert_allclose(np.linalg.inv(Jl), Jl_inv)

    def test_right_jacobian_of_transformation(self):
        for i in range(100):
            T = SE2.random()
            v = np.random.uniform(-10, 10, size=2)

            vp, Jr = T.transa(v, Jr=np.eye(3))
            vx = np.array([-v[1], v[0]])
            Jr_true = np.block([T.R, (T.R @ vx)[:,None]])

            np.testing.assert_allclose(Jr_true, Jr)

    def test_left_jacobian_of_transformation(self):
        for i in range(100):
            T = SE2.random()
            v = np.random.uniform(-10, 10, size=2)

            vp, Jl = T.transa(v, Jl=np.eye(3))
            _, Jr = T.transa(v, Jr=np.eye(3))

            Jl_true = np.eye(2) @ Jr @ np.linalg.inv(T.Adj)

            np.testing.assert_allclose(Jl_true, Jl, atol=1e-10)

    def test_jacobians_of_composition_second_element(self):
        for i in range(100):
            T1 = SE2.random()
            T2 = SE2.random()

            T3, Jr2 = T1.compose(T2, Jr2=np.eye(3))
            _, Jl2 = T1.compose(T2, Jl2=np.eye(3))

            Jl2_true = T3.Adj @ Jr2 @ np.linalg.inv(T2.Adj)

            np.testing.assert_allclose(Jl2_true, Jl2)

    def test_right_jacobian_or_transp(self):
        for i in range(100):
            T = SE2.random()
            v = np.random.uniform(-10, 10, size=2)

            vp, Jr = T.transp(v, Jr=np.eye(3))
            one_x = np.array([[0, -1], [1, 0]])
            Jr_true = np.block([np.eye(2), (one_x @ T.R.T @ (T.t - v))[:,None]])

    def test_left_jacobian_of_transp(self):
        for i in range(100):
            T = SE2.random()
            v = np.random.uniform(-10, 10, size=2)

            vp, Jl = T.transp(v, Jl=np.eye(3))
            one_x = np.array([[0, -1], [1, 0]])
            Jl_true = np.block([-T.R.T, -(T.R.T @ one_x @ v)[:,None]])

            np.testing.assert_allclose(Jl_true, Jl)

    def test_jacobians_of_boxplusr(self):
        for i in range(100):
            T = SE2.random()
            theta = np.random.uniform(-np.pi, np.pi)
            p = np.random.uniform(-10, 10, size=2)
            tau = np.array([*p, theta])

            T2, Jr = T.boxplusr(tau, Jr=np.eye(3))
            _, Jl = T.boxplusr(tau, Jl=np.eye(3))

            Jl_true = T2.Adj @ Jr @ np.eye(3)

            np.testing.assert_allclose(Jl_true, Jl)

    def test_right_jacobians_of_boxminusr(self):
        for i in range(100):
            T1, T2 = SE2.random(), SE2.random()

            tau, Jr1 = T1.boxminusr(T2, Jr1=np.eye(3))
            dT = T2.inv() * T1
            _, Jr1_true = SE2.Log(dT, Jr=np.eye(3))

            _, Jr2 = T1.boxminusr(T2, Jr2=np.eye(3))
            _, Jr2_true = SE2.Log(dT, Jl=np.eye(3))

            np.testing.assert_allclose(Jr1_true, Jr1)
            np.testing.assert_allclose(-Jr2_true, Jr2)

    def test_left_jacobians_of_boxminusr(self):
        for i in range(100):
            T1, T2 = SE2.random(), SE2.random()

            tau, Jl1 = T1.boxminusr(T2, Jl1=np.eye(3))
            _, Jr1 = T1.boxminusr(T2, Jr1=np.eye(3))
            Jl1_true = np.eye(3) @ Jr1 @ np.linalg.inv(T1.Adj)

            _, Jl2 = T1.boxminusr(T2, Jl2=np.eye(3))
            _, Jr2 = T1.boxminusr(T2, Jr2=np.eye(3))
            Jl2_true = np.eye(3) @ Jr2 @ np.linalg.inv(T2.Adj)

            np.testing.assert_allclose(Jl1_true, Jl1)
            np.testing.assert_allclose(Jl2_true, Jl2)

    def test_jacobians_for_boxplusl(self):
        for i in range(100):
            T1 = SE2.random()
            t = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)
            tau = np.array([*t, theta])

            T2, Jr = T1.boxplusl(tau, Jr=np.eye(3))
            _, Jl = T1.boxplusl(tau, Jl=np.eye(3))

            Jl_true = T2.Adj @ Jr @ np.eye(3)
            np.testing.assert_allclose(Jl_true, Jl)

    def test_jacobians_of_boxminusl(self):
        for i in range(100):
            T1, T2 = SE2.random(), SE2.random()

            tau, Jr = T1.boxminusl(T2, Jr1=np.eye(3))
            _, Jl = T1.boxminusl(T2, Jl1=np.eye(3))

            Jl_true = np.eye(3) @ Jr @ np.linalg.inv(T1.Adj)
            np.testing.assert_allclose(Jl_true, Jl)

    def test_jacobians_of_boxminusl_second_element(self):
        for i in range(100):
            T1, T2 = SE2.random(), SE2.random()

            tau, Jr2 = T1.boxminusl(T2, Jr2=np.eye(3))
            _, Jl2 = T1.boxminusl(T2, Jl2=np.eye(3))

            Jl_true = np.eye(3) @ Jr2 @ np.linalg.inv(T2.Adj)
            np.testing.assert_allclose(Jl_true, Jl2)


if __name__=="__main__":
    unittest.main()
