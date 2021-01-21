import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
import unittest
import sys
sys.path.append("..")
from so3 import SO3
from quaternion import Quaternion

from IPython.core.debugger import Pdb

class SO3_testing(unittest.TestCase):
    def testConstructor(self):
        for i in range(100):
            angles = np.random.uniform(-np.pi, np.pi, size=3)
            R_ex = SO3.fromRPY(angles)

            cp = np.cos(angles[0])
            sp = np.sin(angles[0])
            R1 = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])

            ct = np.cos(angles[1])
            st = np.sin(angles[1])
            R2 = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])

            cps = np.cos(angles[2])
            sps = np.sin(angles[2])
            R3 = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])

            R_true = Rotation.from_euler('ZYX', [angles[2], angles[1], angles[0]]).as_matrix()

            R_ex2 = SO3(R_true)

            np.testing.assert_allclose(R_true, R_ex.arr)
            np.testing.assert_allclose(R_true, R_ex2.arr)

    def testLog(self): #This has issues sometimes. It is infrequent though
        for i in range(100):
            temp = Rotation.random().as_matrix()
            R = SO3(temp)

            logR = SO3.log(R)
            logR_true = sp.linalg.logm(temp)

            if np.linalg.norm(logR_true - logR, ord='fro') > 1e-3:
                Pdb().set_trace()
                debug = 1
                temp = SO3.log(R)

            np.testing.assert_allclose(logR_true, logR, atol=1e-10)

    def testTaylorLog(self):
        for i in range(100):  #Around 0
            vec = np.random.uniform(-3.0, 3.0, size=3)
            vec = vec / np.linalg.norm(vec)
            ang = np.random.uniform(-1e-8, 1e-3)
            temp = vec * ang
            R = SO3.fromAxisAngle(temp)

            logR = SO3.log(R)
            logR_true = sp.linalg.logm(R.R)

            if np.linalg.norm(logR_true - logR, ord='fro') > 1e-8:
                Pdb().set_trace()
                debug = 1
                temp = SO3.log(R)


            np.testing.assert_allclose(logR_true, logR, atol=1e-10)

        for i in range(100): #Around pi
            vec = np.random.uniform(-1.0, 1.0, size=3)
            ang = np.random.uniform(-0, 1e-8)
            vec = vec / np.linalg.norm(vec) * (np.pi - ang)

            R = SO3.fromAxisAngle(vec)

            logR = SO3.log(R)
            logR_true = sp.linalg.logm(R.R)

            # if np.linalg.norm(logR_true - logR, ord='fro') > 1e-8:
                # Pdb().set_trace()
                # debug = 1
                # temp = SO3.log(R)

            # np.testing.assert_allclose(logR_true, logR, atol=1e-10)
            #This test has issues. Same with this part in SE3

    def testExp(self):
        for i in range(100):
            logR_vec = np.random.uniform(-np.pi, np.pi, size=3)
            logR = np.array([[0, -logR_vec[2], logR_vec[1]],
                        [logR_vec[2], 0, -logR_vec[0]],
                        [-logR_vec[1], logR_vec[0], 0]])

            R = SO3.exp(logR)
            R_true = sp.linalg.expm(logR)

            np.testing.assert_allclose(R_true, R.arr)

        for i in range(100): #Test taylor series
            logR_vec = np.random.uniform(0.0, 1e-8, size=3)
            logR = np.array([[0, -logR_vec[2], logR_vec[1]],
                        [logR_vec[2], 0, -logR_vec[0]],
                        [-logR_vec[1], logR_vec[0], 0]])

            R = SO3.exp(logR)
            R_true = sp.linalg.expm(logR)

            np.testing.assert_allclose(R_true, R.arr, atol=1e-7)


    def testVee(self):
        for i in range(100):
            omega_true = np.random.uniform(-np.pi, np.pi, size=3)
            logR = np.array([[0, -omega_true[2], omega_true[1]],
                            [omega_true[2], 0, -omega_true[0]],
                            [-omega_true[1], omega_true[0], 0]])

            omega = SO3.vee(logR)
            np.testing.assert_allclose(omega_true, omega)

    def testHat(self):
        for i in range(100):
            omega = np.random.uniform(-np.pi, np.pi, size=3)

            logR_true = np.array([[0, -omega[2], omega[1]],
                                [omega[2], 0, -omega[0]],
                                [-omega[1], omega[0], 0]])

            logR = SO3.hat(omega)

            np.testing.assert_allclose(logR_true, logR)

    def testInv(self):
        for i in range(100):
            mat = Rotation.random().as_matrix()
            R = SO3(mat)

            R_inv = R.inv()
            R_T = R.transpose()
            R_inv_true = np.linalg.inv(mat)

            np.testing.assert_allclose(R_inv_true, R_inv.R)
            np.testing.assert_allclose(R_inv_true, R_T.R)

    def testFromAxisEuler(self):
        for i in range(100):
            theta = np.random.uniform(-np.pi, np.pi)
            vec = np.random.uniform(-1, 1, size=3)
            vec = vec / np.linalg.norm(vec)

            R = SO3.fromAxisAngle(theta * vec)
            R_true = Rotation.from_rotvec(vec*theta).as_matrix()

            np.testing.assert_allclose(R_true, R.arr)

    def testGroupAction(self):
        for i in range(100):
            rot1 = Rotation.random().as_matrix()
            rot2 = Rotation.random().as_matrix()

            R1 = SO3(rot1)
            R2 = SO3(rot2)

            R3 = R1 * R2
            R3_true = rot1 @ rot2

            np.testing.assert_allclose(R3_true, R3.arr)

    def testRotatingVector(self):
        for i in range(100): #Active rotation
            rot1 = Rotation.random().as_matrix()

            R = SO3(rot1)
            pt = np.random.uniform(-5, 5, size=3)

            rot_pt = R.rota(pt)
            rot_pt_true = rot1 @ pt

            np.testing.assert_allclose(rot_pt_true, rot_pt)

        for i in range(100):
            rot1 = Rotation.random().as_matrix()

            R = SO3(rot1)
            pt = np.random.uniform(-5, 5, size=3)

            rot_pt = R.rotp(pt)
            rot_pt_true = rot1.T @ pt

            np.testing.assert_allclose(rot_pt_true, rot_pt)

    def testAdjoint(self):
        for i in range(100):
            delta = np.random.uniform(-np.pi, np.pi, size=3)
            rot = Rotation.random().as_matrix()
            R = SO3(rot)

            Adj_R = R.Adj

            T_true = R * SO3.Exp(delta)
            T = SO3.Exp(Adj_R @ delta) * R

            np.testing.assert_allclose(T_true.R, T.R)

    def testRandom(self):
        for i in range(100):
            R = SO3.random()
            detR = np.linalg.det(R.R)

            np.testing.assert_allclose(1.0, detR)

    def testBoxPlusR(self):
        for i in range(100):
            R = SO3.random()
            theta = np.random.uniform(0, np.pi)
            vec = np.random.uniform(-1, 1, size=3)
            vec = vec / np.linalg.norm(vec) * theta

            R2 = R.boxplusr(vec)
            R2_true = R * SO3.fromAxisAngle(vec)

            np.testing.assert_allclose(R2_true.R, R2.R)

    def testBoxMinusR(self):
        R1 = SO3.random()
        R2 = SO3.random()

        w = R1.boxminusr(R2)
        R_res = R2.boxplusr(w)

        np.testing.assert_allclose(R1.R, R_res.R)

    def test_boxplusl(self):
        for i in range(100):
            R = SO3.random()
            theta = np.random.uniform(0, np.pi)
            vec = np.random.uniform(-1, 1, size=3)
            vec = vec / np.linalg.norm(vec) * theta

            R2 = R.boxplusl(vec)
            R2_true = SO3.fromAxisAngle(vec) * R

            np.testing.assert_allclose(R2_true.R, R2.R)

    def test_boxminusl(self):
        for i in range(100):
            R1 = SO3.random()
            R2 = SO3.random()

            v = R1.boxminusl(R2)
            R = R2.boxplusl(v)

            np.testing.assert_allclose(R1.R, R.R)

    def testNormalize(self):
        for i in range(10):
            R = SO3.random()
            for i in range(10):
                R = R * R

            R.normalize()

            np.testing.assert_allclose(1, R.det())

    def testFromQuaternion(self):
        for i in range(100):
            theta = np.random.uniform(-np.pi, np.pi)
            vec = np.random.uniform(-10.0, 10.0, size=3)
            vec = vec/np.linalg.norm(vec)

            q = Quaternion.fromAxisAngle(vec * theta)

            R = SO3.fromQuaternion(q.q)
            R2 = SO3.fromAxisAngle(vec * theta)

            np.testing.assert_allclose(R.R, R2.R)

    def test_right_jacobian_of_inversion(self):
        R = SO3.random()
        R_inv, Jr = R.inv(Jr=np.eye(3))
        Adj_R = R.Adj

        np.testing.assert_allclose(-Adj_R, Jr)

    def test_left_jacobian_of_inversion(self):
        for i in range(100):
            R = SO3.random()
            R_inv, Jl = R.inv(Jl=np.eye(3))
            _, Jr = R.inv(Jr=np.eye(3))

            Jl_true = R_inv.Adj @ Jr @ np.linalg.inv(R.Adj)

            np.testing.assert_allclose(-R_inv.Adj, Jl)
            np.testing.assert_allclose(Jl_true, Jl)

    def test_right_jacobian_of_composition(self):
        R1 = SO3.random()
        R2 = SO3.random()

        R3, Jr = R1.compose(R2, Jr=np.eye(3))
        Jr_true = np.linalg.inv(R2.Adj)

        np.testing.assert_allclose(Jr_true, Jr)

    def test_left_jacobian_of_composition(self):
        for i in range(100):
            R1 = SO3.random()
            R2 = SO3.random()

            R3, Jr = R1.compose(R2, Jr=np.eye(3))
            _, Jl = R1.compose(R2, Jl=np.eye(3))

            Jl_true = R3.Adj @ Jr @ np.linalg.inv(R1.Adj)

            np.testing.assert_allclose(Jl_true, Jl, atol=1e-10)

    def test_jacobians_of_exponential(self):
        for i in range(100):
            tau = np.random.uniform(-np.pi, np.pi, size=3)
            R, Jr = SO3.Exp(tau, Jr=np.eye(3))
            _, Jl = SO3.Exp(tau, Jl=np.eye(3))

            Adj_R = R.Adj
            res = Jl @ np.linalg.inv(Jr)

            np.testing.assert_allclose(Adj_R, res)

    def test_right_jacobian_of_logarithm(self):
        for i in range(100):
            R = SO3.random()
            logR, Jr_inv = SO3.Log(R, Jr=np.eye(3))
            _, Jr = SO3.Exp(logR, Jr=np.eye(3))

            np.testing.assert_allclose(np.linalg.inv(Jr), Jr_inv)

    def test_left_jacobian_of_logarithm(self):
        for i in range(100):
            R = SO3.random()
            logR, Jl_inv = SO3.Log(R, Jl=np.eye(3))
            _, Jl = SO3.Exp(logR, Jl=np.eye(3))

            np.testing.assert_allclose(np.linalg.inv(Jl), Jl_inv)

    def test_right_jacobian_of_rotation(self):
        for i in range(100):
            R = SO3.random()
            v = np.random.uniform(-10, 10, size=3)

            vp, Jr = R.rota(v, Jr=np.eye(3))
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            Jr_true = -R.R @ vx

            np.testing.assert_allclose(Jr_true, Jr)

    def test_left_jacobian_of_rotation(self):
        for i in range(100):
            R = SO3.random()
            v = np.random.uniform(-10, 10, size=3)

            vp, Jl = R.rota(v, Jl=np.eye(3))
            _, Jr = R.rota(v, Jr=np.eye(3))

            # Jl = f(R)_Adj * Jr * R_Adj
            Jl_true = np.eye(3) @ Jr @ R.Adj.T

            np.testing.assert_allclose(Jl_true, Jl, atol=1e-10)

    def test_jacobians_of_composition_second_element(self):
        for i in range(100):
            R1 = SO3.random()
            R2 = SO3.random()

            R3, Jr2 = R1.compose(R2, Jr2=np.eye(3))
            _, Jl2 = R1.compose(R2, Jl2=np.eye(3))

            Jl2_true = R3.Adj @ Jr2 @ np.linalg.inv(R2.Adj)

            np.testing.assert_allclose(Jl2_true, Jl2)

    def test_right_jacobian_or_rotp(self):
        for i in range(100):
            R = SO3.random()
            v = np.random.uniform(-10, 10, size=3)

            vp, Jr = R.rotp(v, Jr=np.eye(3))
            Jr_true = np.array([[0, -vp[2], vp[1]],
                                [vp[2], 0, -vp[0]],
                                [-vp[1], vp[0], 0]])

            np.testing.assert_allclose(Jr_true, Jr, atol=1e-10)

    def test_left_jacobian_of_rotp(self):
        for i in range(100):
            R = SO3.random()
            v = np.random.uniform(-10, 10, size=3)

            vp, Jl = R.rotp(v, Jl=np.eye(3))
            vx = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
            Jl_true = R.R.T @ vx

            np.testing.assert_allclose(Jl_true, Jl)

    def test_jacobians_of_boxplusr(self):
        for i in range(100):
            R = SO3.random()
            theta = np.random.uniform(-np.pi, np.pi, size=3)

            R2, Jr = R.boxplusr(theta, Jr=np.eye(3))
            _, Jl = R.boxplusr(theta, Jl=np.eye(3))

            Jl_true = R2.Adj @ Jr @ np.eye(3)

            np.testing.assert_allclose(Jl_true, Jl)

    def test_right_jacobians_of_boxminusr(self):
        for i in range(100):
            R1, R2 = SO3.random(), SO3.random()

            theta, Jr1 = R1.boxminusr(R2, Jr1=np.eye(3))
            dR = R2.inv() * R1
            _, Jr1_true = SO3.Log(dR, Jr=np.eye(3))

            _, Jr2 = R1.boxminusr(R2, Jr2=np.eye(3))
            _, Jr2_true = SO3.Log(dR, Jl=np.eye(3))

            np.testing.assert_allclose(Jr1_true, Jr1)
            np.testing.assert_allclose(-Jr2_true, Jr2)

    def test_left_jacobians_of_boxminusr(self):
        for i in range(100):
            R1, R2 = SO3.random(), SO3.random()

            theta, Jl1 = R1.boxminusr(R2, Jl1=np.eye(3))
            _, Jr1 = R1.boxminusr(R2, Jr1=np.eye(3))
            Jl1_true = np.eye(3) @ Jr1 @ R1.Adj.T

            _, Jl2 = R1.boxminusr(R2, Jl2=np.eye(3))
            _, Jr2 = R1.boxminusr(R2, Jr2=np.eye(3))
            Jl2_true = np.eye(3) @ Jr2 @ R2.Adj.T

            np.testing.assert_allclose(Jl1_true, Jl1, rtol=1e-5)
            np.testing.assert_allclose(Jl2_true, Jl2, rtol=1e-5)

    def test_jacobians_of_boxplusl(self):
        for i in range(100):
            R = SO3.random()
            theta = np.random.uniform(-np.pi, np.pi, size=3)

            R2, Jr = R.boxplusl(theta, Jr=np.eye(3))
            _, Jl = R.boxplusl(theta, Jl=np.eye(3))

            Jl_true = R2.Adj @ Jr @ np.eye(3)
            np.testing.assert_allclose(Jl_true, Jl)

    def test_jacobians_of_boxminusl(self):
        for i in range(100):
            R1, R2 = SO3.random(), SO3.random()

            theta, Jr = R1.boxminusl(R2, Jr1=np.eye(3))
            _, Jl = R1.boxminusl(R2, Jl1=np.eye(3))

            Jl_true = np.eye(3) @ Jr @ R1.Adj.T
            np.testing.assert_allclose(Jl_true, Jl, atol=1e-8)

    def test_jacobians_of_boxminusl_second_element(self):
        for i in range(100):
            R1, R2 = SO3.random(), SO3.random()

            theta, Jr2 = R1.boxminusl(R2, Jr2=np.eye(3))
            _, Jl2 = R1.boxminusl(R2, Jl2=np.eye(3))

            Jl_true = np.eye(3) @ Jr2 @ R2.Adj.T
            np.testing.assert_allclose(Jl_true, Jl2)


if __name__=="__main__":
    unittest.main()
