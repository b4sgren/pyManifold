import unittest
import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
import sys

from test_quaternion import quatMultiply
sys.path.append('..')
from se3 import SE3
from so3 import SO3
from quaternion import Quaternion, skew

from IPython.core.debugger import Pdb

class SE3_Test(unittest.TestCase):
    def setUp(self):
        self.transforms = [SE3.random() for i in range(100)]

    def test_random_generator(self):
        for T in self.transforms:
            is_valid = T.isValidTransform()
            self.assertTrue(is_valid)

    def test_rotation_from_quaternion(self):
        for T in self.transforms:
            R = T.R # because from quaternion
            R_true = SO3.fromQuaternion(T.q_arr).R

            np.testing.assert_allclose(R_true, R)

    def test_composition(self):
        transforms2 = [SE3.random() for i in range(100)]
        for (T1, T2) in zip(self.transforms, transforms2):
            t1 = T1.t
            t2 = T2.t

            T3 = T1 * T2

            q3_true = quatMultiply(T1.q, T2.q)
            if q3_true[0] < 0:
                q3_true *= -1

            t3_true = T1.t + T1.R @ T2.t
            T3_true = SE3(Quaternion(q3_true), t3_true)

            np.testing.assert_allclose(T3_true.q_arr, T3.q_arr)
            np.testing.assert_allclose(T3_true.t, T3.t)

    # def test_inverse(self):
    #     for T in self.transforms:
    #         T_inv = T.inv()

    #         I = T_inv * T

    #         np.testing.assert_allclose(I.q_arr, np.array([1, 0, 0, 0]))
    #         np.testing.assert_allclose(I.t, np.zeros(3))

    # def test_from_rot_and_trans(self):
    #     rots = [SO3.random().R for i in range(100)]
    #     trans = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     for (R,t) in zip(rots, trans):
    #         T = SE3.fromRAndt(R,t)

    #         q_true = Quaternion.fromRotationMatrix(R)

    #         np.testing.assert_allclose(q_true.q, T.q_arr)
    #         np.testing.assert_allclose(t, T.t)

    # def test_transforma(self):
    #     pt = np.array([1, 0, 0])
    #     T = SE3.fromAxisAngleAndt(np.pi/2 * np.array([0, 0, 1]), np.array([0, 1, 0]))
    #     pt_p = T.transa(pt)
    #     pt_true = np.array([0, 2, 0])
    #     np.testing.assert_allclose(pt_true, pt_p, atol=1e-10)

    # def test_transformp(self):
    #     pt = np.array([1, 0, 0])
    #     T = SE3.fromAxisAngleAndt(np.pi/2 * np.array([0, 0, 1]), np.array([0, 1, 0]))
    #     pt_p = T.transp(pt)
    #     pt_true = np.array([-1, -1, 0])
    #     np.testing.assert_allclose(pt_true, pt_p, atol=1e-10)

    # def test_active_transforming_a_point(self):
    #     pts = [np.random.uniform(-3.0, 3.0, size=3) for i in range(100)]
    #     for (T,pt) in zip(self.transforms, pts):
    #         pt_p = T.transa(pt)

    #         pt_p_true = T.t + T.R.T @ pt

    #         np.testing.assert_allclose(pt_p_true, pt_p)

    # def test_passive_transforming_a_point(self):
    #     pts = [np.random.uniform(-3.0, 3.0, size=3) for i in range(100)]
    #     for(T,pt) in zip(self.transforms, pts):
    #         pt_p = T.transp(pt)

    #         pt_p_true = -T.R @ T.t + T.R @ pt

    #         np.testing.assert_allclose(pt_p_true, pt_p)

    # def test_from_RPY_and_trans(self):
    #     rpy = [np.random.uniform(-np.pi, np.pi, size=3) for i in range(100)]
    #     trans = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     for (eul, t) in zip(rpy, trans):
    #         T = SE3.fromRPYandt(eul, t)

    #         R = SO3.fromRPY(eul).R
    #         # q = Quaternion.fromRotationMatrix(R.T)
    #         q = Quaternion.fromRotationMatrix(R)

    #         np.testing.assert_allclose(q.q, T.q_arr)
    #         np.testing.assert_allclose(t, T.t)

    # def test_from_axis_angle(self):
    #     angle = [np.random.uniform(0.0, np.pi) for i in range(100)]
    #     axis = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     trans = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     for (v, phi, t) in zip(axis, angle, trans):
    #         v = v / np.linalg.norm(v) * phi
    #         T = SE3.fromAxisAngleAndt(v,t)

    #         q = Quaternion.fromAxisAngle(v)

    #         np.testing.assert_allclose(q.q, T.q_arr)
    #         np.testing.assert_allclose(t, T.t)

    # def test_from_axis_angle_taylor_series(self):
    #     angle = [np.random.uniform(0.0, 1e-3) for i in range(100)]
    #     axis = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     trans = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     for (v, phi, t) in zip(axis, angle, trans):
    #         v = v / np.linalg.norm(v) * phi
    #         T = SE3.fromAxisAngleAndt(v,t)

    #         q = Quaternion.fromAxisAngle(v)

    #         np.testing.assert_allclose(q.q, T.q_arr)
    #         np.testing.assert_allclose(t, T.t)

    # def test_from_7_vec(self):
    #     for T in self.transforms:
    #         arr = np.array([*T.t, *T.q_arr])
    #         T2 = SE3.from7vec(arr)

    #         np.testing.assert_allclose(T.q_arr, T2.q_arr)
    #         np.testing.assert_allclose(T.t, T2.t)

    # def test_hat(self):
    #     algebras = [np.random.uniform(-3.0, 3.0, size=6) for i in range(100)]
    #     for vec in algebras:
    #         logT = SE3.hat(vec)
    #         logT_true = np.array([vec[0], vec[1], vec[2], 0, vec[3], vec[4], vec[5]])

    #         np.testing.assert_allclose(logT_true, logT)

    # def test_vee(self):
    #     vec = [np.random.uniform(-3.0, 3.0, size=7) for i in range(100)]
    #     for v in vec:
    #         res = SE3.vee(v)
    #         res_true = np.array([v[0], v[1], v[2], v[4], v[5], v[6]])

    #         np.testing.assert_allclose(res_true, res)

    # def test_logarithmic_map(self):
    #     for T in self.transforms:
    #         logT = SE3.log(T)

    #         R = T.R.T
    #         t = T.t
    #         T2 = np.block([[R, t[:,None]], [np.zeros((1,3)), 1]])
    #         temp = sp.linalg.logm(T2)
    #         logT_true = np.array([temp[0,3], temp[1,3], temp[2,3], 0, temp[2,1], temp[0,2], temp[1,0]])

    #         np.testing.assert_allclose(logT_true, logT)

    # def test_logarithmic_map_taylor_series(self):
    #     angle = [np.random.uniform(0.0, 1e-8) for i in range(100)]
    #     axis = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     trans = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     for (v, phi, t) in zip(axis, angle, trans):
    #         v = v / np.linalg.norm(v) * phi
    #         T = SE3.fromAxisAngleAndt(v,t)

    #         logT = SE3.log(T)

    #         R = T.R.T
    #         t = T.t
    #         T2 = np.block([[R, t[:,None]], [np.zeros((1,3)), 1]])
    #         temp = sp.linalg.logm(T2)
    #         logT_true = np.array([temp[0,3], temp[1,3], temp[2,3], 0, temp[2,1], temp[0,2], temp[1,0]])

    #         np.testing.assert_allclose(logT_true, logT, atol=1e-3, rtol=1e-3)

    # def test_exponential_map(self):
    #     v_list = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     w_list = [np.random.uniform(-np.pi, np.pi, size=3) for i in range(100)]
    #     for (v,w) in zip(v_list, w_list):
    #         vec = np.array([*v, *w])
    #         T = SE3.Exp(vec)

    #         logT = np.array([[0, -w[2], w[1], v[0]],
    #                          [w[2], 0, -w[0], v[1]],
    #                          [-w[1], w[0], 0, v[2]],
    #                          [0, 0, 0, 0]])
    #         T_true = sp.linalg.expm(logT)
    #         R_true = T_true[:3,:3]
    #         t_true = T_true[:3,3]

    #         np.testing.assert_allclose(R_true, T.R.T)
    #         np.testing.assert_allclose(t_true, T.t)

    # def test_exponential_map_taylor_series(self):
    #     axis = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     v_list = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     angles = [np.random.uniform(0, 1e-8) for i in range(100)]
    #     for (a,v,psi) in zip(axis, v_list, angles):
    #         w = a / np.linalg.norm(a) * psi
    #         vec = np.array([*v, *w])

    #         T = SE3.Exp(vec)

    #         logT = np.array([[0, -w[2], w[1], v[0]],
    #                          [w[2], 0, -w[0], v[1]],
    #                          [-w[1], w[0], 0, v[2]],
    #                          [0, 0, 0, 0]])
    #         T_true = sp.linalg.expm(logT)
    #         R_true = T_true[:3,:3]
    #         t_true = T_true[:3,3]

    #         np.testing.assert_allclose(R_true, T.R.T, rtol=1e-4, atol=1e-4)
    #         np.testing.assert_allclose(t_true, T.t, rtol=1e-4, atol=1e-4)

    # def test_adjoint(self):
    #     v_list = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     w_list = [np.random.uniform(-np.pi, np.pi, size=3) for i in range(100)]
    #     for (v,w, T) in zip(v_list, w_list, self.transforms):
    #         vec = np.array([*v, *w])

    #         T1 = T * SE3.Exp(vec)
    #         T2 = SE3.Exp(T.Adj @ vec) * T

    #         np.testing.assert_allclose(T1.T, T2.T)

    # def test_normalize(self):
    #     T = SE3.Identity()
    #     T2 = self.transforms[30]
    #     for i in range(1000):
    #         T = T * T2
    #     T.normalize()
    #     self.assertTrue(T.isValidTransform())

    # def test_boxplusr(self):
    #     v_list = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     w_list = [np.random.uniform(-np.pi, np.pi, size=3) for i in range(100)]
    #     for (T, v, w) in zip(self.transforms, v_list, w_list):
    #         vec = np.array([*v, *w])
    #         T2 = T.boxplusr(vec)

    #         temp = SE3.Exp(vec)
    #         T2_true = T * temp

    #         np.testing.assert_allclose(T2_true.T, T2.T)

    # def test_boxminusr(self):
    #     transforms2 = [SE3.random() for i in range(100)]
    #     for (T1, T2) in zip(self.transforms, transforms2):
    #         diffT = T1.boxminusr(T2)
    #         T = T2.boxplusr(diffT)

    #         np.testing.assert_allclose(T.T, T1.T)

    # def test_boxplusl(self):
    #     v_list = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
    #     w_list = [np.random.uniform(-np.pi, np.pi, size=3) for i in range(100)]
    #     for (T,v,w) in zip(self.transforms, v_list, w_list):
    #         vec = np.array([*v, *w])
    #         T2 = T.boxplusl(vec)

    #         T2_true = SE3.Exp(vec) * T

    #         np.testing.assert_allclose(T2_true.T, T2.T)

    # def test_boxminusl(self):
    #     transforms2 = [SE3.random() for i in range(100)]
    #     for (T1,T2) in zip(self.transforms, transforms2):
    #         v = T1.boxminusl(T2)
    #         T = T2.boxplusl(v)

    #         np.testing.assert_allclose(T.T, T1.T)

    # def test_right_jacobian_of_inversion(self):
    #     T = SE3.random()
    #     T_inv, Jr = T.inv(Jr=np.eye(6))

    #     np.testing.assert_allclose(-T.Adj, Jr)

    # def test_left_jacobian_of_inversion(self):
    #     T = SE3.random()
    #     T_inv, Jr = T.inv(Jr=np.eye(6))
    #     _, Jl = T.inv(Jl=np.eye(6))

    #     Adj_T = T.Adj
    #     Adj_Tinv = T_inv.Adj

    #     Jl_true = Adj_Tinv @ Jr @ np.linalg.inv(Adj_T)

    #     np.testing.assert_allclose(Jl_true, Jl)

    # def test_right_jacobian_of_composition(self):
    #     T1 = SE3.random()
    #     T2 = SE3.random()

    #     T3, Jr = T1.compose(T2, Jr=np.eye(6))
    #     Jr_true = np.linalg.inv(T2.Adj)

    #     np.testing.assert_allclose(Jr_true, Jr)

    # def test_left_jacobian_of_composition(self):
    #     for i in range(100):
    #         T1 = SE3.random()
    #         T2 = SE3.random()

    #         T3, Jr = T1.compose(T2, Jr=np.eye(6))
    #         _, Jl = T1.compose(T2, Jl=np.eye(6))

    #         Jl_true = T3.Adj @ Jr @ T1.inv().Adj

    #         np.testing.assert_allclose(Jl_true, Jl, atol=1e-10)

    # def test_jacobians_of_exponential(self):
    #     for i in range(100):
    #         rho = np.random.uniform(-10, 10, size=3)
    #         theta = np.random.uniform(-np.pi, np.pi, size=3)
    #         tau = np.array([*rho, *theta])

    #         T, Jr = SE3.Exp(tau, Jr=np.eye(6))
    #         _, Jl = SE3.Exp(-tau, Jl=np.eye(6))

    #         np.testing.assert_allclose(Jr, Jl)

    # def test_left_jacobian_of_logarithm(self):
    #     for i in range(100):
    #         T = SE3.random()
    #         logT, Jl_inv = SE3.Log(T, Jl=np.eye(6))
    #         _, Jl = SE3.Exp(logT, Jl=np.eye(6))

    #         np.testing.assert_allclose(np.linalg.inv(Jl), Jl_inv)

    # def test_right_jacobian_of_logarithm(self):
    #     for i in range(100):
    #         T = SE3.random()
    #         logT, Jr_inv = SE3.Log(T, Jr=np.eye(6))
    #         _, Jr = SE3.Exp(logT, Jr=np.eye(6))

    #         np.testing.assert_allclose(np.linalg.inv(Jr), Jr_inv)

    # def test_right_jacobian_of_transformation(self):
    #     for i in range(100):
    #         T = SE3.random()
    #         v = np.random.uniform(-10, 10, size=3)

    #         vp, Jr = T.transa(v, Jr=np.eye(6))
    #         vx = np.array([[0, -v[2], v[1]],
    #                        [v[2], 0, -v[0]],
    #                        [-v[1], v[0], 0]])
    #         Jr_true = np.block([T.R.T, -T.R.T @ vx])

    #         np.testing.assert_allclose(Jr_true, Jr)

    # def test_left_jacobian_of_transformation(self):
    #     for i in range(100):
    #         T = SE3.random()
    #         v = np.random.uniform(-10, 10, size=3)

    #         vp, Jl = T.transa(v, Jl=np.eye(6))
    #         _, Jr = T.transa(v, Jr=np.eye(6))

    #         Jl_true = np.eye(3) @ Jr @ np.linalg.inv(T.Adj)

    #         np.testing.assert_allclose(Jl_true, Jl, atol=1e-10)

    # def test_jacobians_of_composition_second_element(self):
    #     for i in range(100):
    #         T1 = SE3.random()
    #         T2 = SE3.random()

    #         T3, Jr2 = T1.compose(T2, Jr2=np.eye(6))
    #         _, Jl2 = T1.compose(T2, Jl2=np.eye(6))

    #         Jl2_true = T3.Adj @ Jr2 @ np.linalg.inv(T2.Adj)

    #         np.testing.assert_allclose(Jl2_true, Jl2)

    # def test_right_jacobian_of_transp(self):
    #     for T in self.transforms:
    #         v = np.random.uniform(-10, 10, size=3)

    #         vp, Jr = T.transp(v, Jr=np.eye(6))
    #         Jr_true = np.block([-np.eye(3), skew(vp)])

    #         np.testing.assert_allclose(Jr_true, Jr, atol=1e-10)

    # def test_left_jacobian_of_transp(self):
    #     for T in self.transforms:
    #         v = np.random.uniform(-10, 10, size=3)

    #         vp, Jl = T.transp(v, Jl=np.eye(6))
    #         vx = skew(v)
    #         Jl_true = np.block([-T.R, T.R @ vx])

    #         np.testing.assert_allclose(Jl_true, Jl)

    # def test_right_jacobian_of_boxplusr(self):
    #     for T in self.transforms:
    #         v = np.random.uniform(-10, 10, size=3)
    #         theta = np.random.uniform(-np.pi, np.pi, size=3)
    #         tau = np.array([*v, *theta])

    #         T2, Jr = T.boxplusr(tau, Jr=np.eye(6))
    #         _, Jr_true = SE3.Exp(tau, Jr=np.eye(6))

    #         np.testing.assert_allclose(Jr_true, Jr)

    # def test_left_jacobian_of_boxplusr(self):
    #     for T in self.transforms:
    #         v = np.random.uniform(-10, 10, size=3)
    #         theta = np.random.uniform(-np.pi, np.pi, size=3)
    #         tau = np.array([*v, *theta])

    #         T2, Jr = T.boxplusr(tau, Jr=np.eye(6))
    #         _, Jl = T.boxplusr(tau, Jl=np.eye(6))

    #         Jl_true = T2.Adj @ Jr @ np.eye(6)

    #         np.testing.assert_allclose(Jl_true, Jl)

    # def test_right_jacobians_of_boxminusr(self):
    #     for T1 in self.transforms:
    #         T2 = SE3.random()

    #         tau, Jr1 = T1.boxminusr(T2, Jr1=np.eye(6))
    #         dT = T2.inv() * T1
    #         _, Jr1_true = SE3.Log(dT, Jr=np.eye(6))

    #         _, Jr2 = T1.boxminusr(T2, Jr2=np.eye(6))
    #         _, Jr2_true = SE3.Log(dT, Jl=np.eye(6))

    #         np.testing.assert_allclose(Jr1_true, Jr1)
    #         np.testing.assert_allclose(-Jr2_true, Jr2)

    # def test_left_jacobians_of_boxminusr(self):
    #     for T1 in self.transforms:
    #         T2 = SE3.random()

    #         tau, Jl1 = T1.boxminusr(T2, Jl1=np.eye(6))
    #         _, Jr1 = T1.boxminusr(T2, Jr1=np.eye(6))
    #         Jl1_true = np.eye(6) @ Jr1 @ np.linalg.inv(T1.Adj)

    #         _, Jl2 = T1.boxminusr(T2, Jl2=np.eye(6))
    #         _, Jr2 = T1.boxminusr(T2, Jr2=np.eye(6))
    #         Jl2_true = np.eye(6) @ Jr2 @ np.linalg.inv(T2.Adj)

    #         np.testing.assert_allclose(Jl1_true, Jl1)
    #         np.testing.assert_allclose(Jl2_true, Jl2)

    # def test_jacobians_of_boxplusl(self):
    #     for T in self.transforms:
    #         t = np.random.uniform(-10, 10, size=3)
    #         theta = np.random.uniform(-np.pi, np.pi, size=3)
    #         tau = np.array([*t, *theta])

    #         T2, Jr = T.boxplusl(tau, Jr=np.eye(6))
    #         T2, Jl = T.boxplusl(tau, Jl=np.eye(6))

    #         Jl_true = T2.Adj @ Jr @ np.eye(6)
    #         np.testing.assert_allclose(Jl_true, Jl)

    # def test_jacobians_of_boxminusl(self):
    #     for T1 in self.transforms:
    #         T2 = SE3.random()

    #         diff, Jr = T1.boxminusl(T2, Jr1=np.eye(6))
    #         diff, Jl = T1.boxminusl(T2, Jl1=np.eye(6))

    #         Jl_true = np.eye(6) @ Jr @ np.linalg.inv(T1.Adj)
    #         np.testing.assert_allclose(Jl_true, Jl)

    # def test_jacobians_of_boxminusl_second_element(self):
    #     for T1 in self.transforms:
    #         T2 = SE3.random()

    #         diff, Jr2 = T1.boxminusl(T2, Jr2=np.eye(6))
    #         diff, Jl2 = T1.boxminusl(T2, Jl2=np.eye(6))

    #         Jl_true = np.eye(6) @ Jr2 @ np.linalg.inv(T2.Adj)
    #         np.testing.assert_allclose(Jl_true, Jl2)

    # def test_matrix(self):
    #     for T1 in self.transforms:
    #         v = np.random.uniform(-10, 10, size=3)
    #         vh = np.concatenate((v, np.array([1])))

    #         vp = T1.transa(v)
    #         vp2 = T1.matrix @ vh

    #         np.testing.assert_allclose(vp, vp2[:-1])

if __name__=="__main__":
    unittest.main()
