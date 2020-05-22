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
            theta = np.random.uniform(-1e-3, 1e-3)

            T = SE2.fromAngleAndt(theta, t)
            logT = SE2.log(T)
            logT_true = spl.logm(T.arr)
            
            if np.linalg.norm(logT_true - logT, ord='fro') > 1e-3:
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
            theta = np.random.uniform(-1e-3, 1e-3)
            arr = np.array([theta, v[0], v[1]])

            T = SE2.Exp(arr)
            T_true = spl.expm(SE2.hat(arr))

            np.testing.assert_allclose(T_true, T.arr)
    
    def testVee(self):
        for i in range(100):
            u = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)

            arr_true = np.array([theta, u[0], u[1]])
            X = np.array([[0, -theta, u[0]], [theta, 0, u[1]], [0, 0, 0]])
            arr = SE2.vee(X)

            np.testing.assert_allclose(arr_true, arr)
    
    def testHat(self):
        for i in range(100):
            u = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)

            arr = np.array([theta, u[0], u[1]])
            X_true = np.array([[0, -theta, u[0]], [theta, 0, u[1]], [0, 0, 0]])
            X = SE2.hat(arr)

            np.testing.assert_allclose(X_true, X)
    
    def testAdjoint(self): 
        for i in range(100):
            t = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)

            ct = np.cos(theta)
            st = np.sin(theta)
            R = np.array([[ct, -st], [st, ct]])
            T = SE2.fromRandt(R, t)
            adj = T.Adj

            adj_true = np.zeros((3,3))
            adj_true[1:, 1:] = R 
            adj_true[1, 0] = t[1]
            adj_true[2, 0] = -t[0]
            adj_true[0,0] = 1

            np.testing.assert_allclose(adj_true, adj)
    
        for i in range(100):
            t = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)
            u = np.random.uniform(-1, 1, size=2)
            phi = np.random.uniform(-np.pi, np.pi)
            delta = np.array([phi, u[0], u[1]])

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
    
    def testBoxPlus(self):
        for i in range(100):
            T = SE2.random()
            u = np.random.uniform(-10, 10, size=2)
            theta = np.random.uniform(-np.pi, np.pi)
            vec = np.array([theta, *u])

            T3 = T.boxplus(vec)
            T3_true = T *  SE2.Exp(vec)

            np.testing.assert_allclose(T3_true.T, T3.T)
    
    def testBoxMinus(self):
        for i in range(100):
            T1 = SE2.random()
            T2 = SE2.random()

            w = T1.boxminus(T2)
            w_true = SE2.Log(T2.inv()*T1)

            np.testing.assert_allclose(w_true, w)


if __name__=="__main__":
    unittest.main()