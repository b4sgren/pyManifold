import unittest
import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
import sys
sys.path.append('..')
from se3 import SE3
from so3 import SO3
from quaternion import Quaternion

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
            R = T.R
            R_true = SO3.fromQuaternion(T.q_arr).R

            np.testing.assert_allclose(R_true, R)

    def test_composition(self):
        transforms2 = [SE3.random() for i in range(100)]
        for (T1, T2) in zip(self.transforms, transforms2):
            t1 = T1.t
            t2 = T2.t

            T3 = T1 * T2

            q3_true = np.array([
                T1.qw * T2.qw - T1.qx * T2.qx - T1.qy * T2.qy - T1.qz * T2.qz,
                T1.qw * T2.qx + T1.qx * T2.qw + T1.qy * T2.qz - T1.qz * T2.qy,
                T1.qw * T2.qy - T1.qx * T2.qz + T1.qy * T2.qw + T1.qz * T2.qx,
                T1.qw * T2.qz + T1.qx * T2.qy - T1.qy * T2.qx + T1.qz * T2.qw])
            if q3_true[0] < 0:
                q3_true *= -1

            t3_true = T1.t + T1.R @ T2.t
            T3_true = SE3(Quaternion(q3_true), t3_true)

            np.testing.assert_allclose(T3_true.q_arr, T3.q_arr)
            np.testing.assert_allclose(T3_true.t, T3.t)

    def test_inverse(self):
        for T in self.transforms:
            T_inv = T.inv()

            I = T_inv * T

            np.testing.assert_allclose(I.q_arr, np.array([1, 0, 0, 0]))
            np.testing.assert_allclose(I.t, np.zeros(3))

    def test_from_rot_and_trans(self):
        rots = [SO3.random().R for i in range(100)]
        trans = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
        for (R,t) in zip(rots, trans):
            T = SE3.fromRAndt(R,t)

            q_true = Quaternion.fromRotationMatrix(R)

            np.testing.assert_allclose(q_true.q, T.q_arr)
            np.testing.assert_allclose(t, T.t)

    def test_active_transforming_a_point(self):
        pts = [np.random.uniform(-3.0, 3.0, size=3) for i in range(100)]
        for (T,pt) in zip(self.transforms, pts):
            pt_p = T.transa(pt)

            pt_p_true = T.t + T.R @ pt

            np.testing.assert_allclose(pt_p_true, pt_p)

    def test_passive_transforming_a_point(self):
        pts = [np.random.uniform(-3.0, 3.0, size=3) for i in range(100)]
        for(T,pt) in zip(self.transforms, pts):
            pt_p = T.transp(pt)

            pt_p_true = -T.R.T @ T.t + T.R.T @ pt

            np.testing.assert_allclose(pt_p_true, pt_p)

    def test_from_RPY_and_trans(self):
        rpy = [np.random.uniform(-np.pi, np.pi, size=3) for i in range(100)]
        trans = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
        for (eul, t) in zip(rpy, trans):
            T = SE3.fromRPYandt(eul, t)

            R = SO3.fromRPY(eul).R
            q = Quaternion.fromRotationMatrix(R)

            np.testing.assert_allclose(q.q, T.q_arr)
            np.testing.assert_allclose(t, T.t)

    def test_from_axis_angle(self):
        angle = [np.random.uniform(0.0, np.pi) for i in range(100)]
        axis = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
        trans = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
        for (v, phi, t) in zip(axis, angle, trans):
            v = v / np.linalg.norm(v) * phi
            T = SE3.fromAxisAngleAndt(v,t)

            q = Quaternion.fromAxisAngle(v)

            np.testing.assert_allclose(q.q, T.q_arr)
            np.testing.assert_allclose(t, T.t)

    def test_from_axis_angle_taylor_series(self):
        angle = [np.random.uniform(0.0, 1e-3) for i in range(100)]
        axis = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
        trans = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
        for (v, phi, t) in zip(axis, angle, trans):
            v = v / np.linalg.norm(v) * phi
            T = SE3.fromAxisAngleAndt(v,t)

            q = Quaternion.fromAxisAngle(v)

            np.testing.assert_allclose(q.q, T.q_arr)
            np.testing.assert_allclose(t, T.t)

    def test_from_7_vec(self):
        for T in self.transforms:
            arr = np.array([*T.t, *T.q_arr])
            T2 = SE3.from7vec(arr)

            np.testing.assert_allclose(T.q_arr, T2.q_arr)
            np.testing.assert_allclose(T.t, T2.t)

    def test_hat(self):
        algebras = [np.random.uniform(-3.0, 3.0, size=6) for i in range(100)]
        for vec in algebras:
            logT = SE3.hat(vec)
            logT_true = np.array([vec[0], vec[1], vec[2], 0, vec[3], vec[4], vec[5]])

            np.testing.assert_allclose(logT_true, logT)

    def test_vee(self):
        vec = [np.random.uniform(-3.0, 3.0, size=7) for i in range(100)]
        for v in vec:
            res = SE3.vee(v)
            res_true = np.array([v[0], v[1], v[2], v[4], v[5], v[6]])

            np.testing.assert_allclose(res_true, res)

    def test_logarithmic_map(self):
        for T in self.transforms:
            logT = SE3.log(T)

            R = T.R
            t = T.t
            T2 = np.block([[R, t[:,None]], [np.zeros((1,3)), 1]])
            temp = sp.linalg.logm(T2)
            logT_true = np.array([temp[0,3], temp[1,3], temp[2,3], 0, temp[2,1], temp[0,2], temp[1,0]])

            np.testing.assert_allclose(logT_true, logT)

    def test_logarithmic_map_taylor_series(self):
        angle = [np.random.uniform(0.0, 1e-8) for i in range(100)]
        axis = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
        trans = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
        for (v, phi, t) in zip(axis, angle, trans):
            v = v / np.linalg.norm(v) * phi
            T = SE3.fromAxisAngleAndt(v,t)

            logT = SE3.log(T)

            R = T.R
            t = T.t
            T2 = np.block([[R, t[:,None]], [np.zeros((1,3)), 1]])
            temp = sp.linalg.logm(T2)
            logT_true = np.array([temp[0,3], temp[1,3], temp[2,3], 0, temp[2,1], temp[0,2], temp[1,0]])

            np.testing.assert_allclose(logT_true, logT, atol=1e-3, rtol=1e-3)

    def test_exponential_map(self):
        v_list = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
        w_list = [np.random.uniform(-np.pi, np.pi, size=3) for i in range(100)]
        for (v,w) in zip(v_list, w_list):
            vec = np.array([*v, *w])
            T = SE3.Exp(vec)

            logT = np.array([[0, -w[2], w[1], v[0]],
                             [w[2], 0, -w[0], v[1]],
                             [-w[1], w[0], 0, v[2]],
                             [0, 0, 0, 0]])
            T_true = sp.linalg.expm(logT)
            R_true = T_true[:3,:3]
            t_true = T_true[:3,3]

            np.testing.assert_allclose(R_true, T.R)
            np.testing.assert_allclose(t_true, T.t)

    def test_exponential_map_taylor_series(self):
        axis = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
        v_list = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
        angles = [np.random.uniform(0, 1e-8) for i in range(100)]
        for (a,v,psi) in zip(axis, v_list, angles):
            w = a / np.linalg.norm(a) * psi
            vec = np.array([*v, *w])

            T = SE3.Exp(vec)

            logT = np.array([[0, -w[2], w[1], v[0]],
                             [w[2], 0, -w[0], v[1]],
                             [-w[1], w[0], 0, v[2]],
                             [0, 0, 0, 0]])
            T_true = sp.linalg.expm(logT)
            R_true = T_true[:3,:3]
            t_true = T_true[:3,3]

            np.testing.assert_allclose(R_true, T.R)
            np.testing.assert_allclose(t_true, T.t, rtol=1e-4, atol=1e-4)

    def test_adjoint(self):
        v_list = [np.random.uniform(-10.0, 10.0, size=3) for i in range(100)]
        w_list = [np.random.uniform(-np.pi, np.pi, size=3) for i in range(100)]
        for (v,w, T) in zip(v_list, w_list, self.transforms):
            vec = np.array([*v, *w])

            T1 = T * SE3.Exp(vec)
            T2 = SE3.Exp(T.Adj @ vec) * T

            np.testing.assert_allclose(T1.T, T2.T)

if __name__=="__main__":
    unittest.main()
