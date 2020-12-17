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


# class SE3_Test(unittest.TestCase):
#     def testConstructor(self):
#         for i in range(100):
#             t = np.random.uniform(-10, 10, size=3)
#             R = Rotation.random().as_matrix()

#             T_true = np.eye(4)
#             T_true[:3,:3] = R
#             T_true[:3,3] = t

#             T = SE3.fromRotationMatrix(t, R)

#             np.testing.assert_allclose(T_true, T.arr)

#     def testLog(self):
#         for i in range(100):
#             t = np.random.uniform(-10, 10, size=3)
#             R = Rotation.random().as_matrix()

#             T = SE3.fromRotationMatrix(t, R)
#             logT = SE3.log(T)
#             logT_true = sp.linalg.logm(T.arr)

#             if np.linalg.norm(logT_true - logT, ord='fro') > 1e-3:
#                 Pdb().set_trace()
#                 debug = 1
#                 logT2 = SE3.log(T)

#             np.testing.assert_allclose(logT_true, logT, atol=1e-8)


#     @unittest.skip("Not stable. Make rotation part a quaternion")
#     def testTaylorLog(self):
#         for i in range(100): #Test taylor series expansion
#             t = np.random.uniform(-10, 10, size=3)
#             ang = np.random.uniform(-1e-3, 1e-3)
#             vec = np.random.uniform(-1.0, 1.0, size=3)
#             vec = vec / np.linalg.norm(vec) * ang

#             R = Rotation.from_rotvec(vec).as_matrix()
#             T = SE3.fromRotationMatrix(t, R)
#             logT = SE3.log(T)
#             logT_true = sp.linalg.logm(T.arr)

#             if np.linalg.norm(logT_true - logT, ord='fro') > 1e-3:
#                 Pdb().set_trace()
#                 debug = 1
#                 logT2 = SE3.log(T)

#             np.testing.assert_allclose(logT_true, logT, atol=1e-8)

#         for i in range(100): #Test taylor series around pi
#             t = np.random.uniform(-10, 10, size=3)
#             ang = np.random.uniform(-1e-3, 1e-3) + np.pi
#             vec = np.random.uniform(-1.0, 1.0, size=3)
#             vec = vec / np.linalg.norm(vec) * ang

#             R = Rotation.from_rotvec(vec).as_matrix()
#             T = SE3.fromRotationMatrix(t, R)
#             logT = SE3.log(T)
#             logT_true = sp.linalg.logm(T.arr)

#             if isinstance(logT_true[0,0], np.complex):
#                 logT_true = np.real(logT_true)

#             if np.linalg.norm(logT_true - logT, ord='fro') > 1e-3:
#                 logT2 = SE3.log(T)

#             np.testing.assert_allclose(logT_true, logT, atol=1e-4, rtol=1e-4) #Values match. For some reason these need to be kinda big
#             #Failure Case: t = array([4.21542429, 9.63179667, 6.94835173]), vec = array([ 2.0673784 , -1.90185657,  1.40659037]), ang = 3.1415932774993163
#             #sp.linalg.logm gives complex values...

#     def testExp(self):
#         for i in range(100):
#             u = np.random.uniform(-10, 10, size=3)
#             w = np.random.uniform(-np.pi, np.pi, size=3)

#             logR = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

#             logT = np.zeros((4,4))
#             logT[:3,:3] = logR
#             logT[:3,3] = u

#             T_true = sp.linalg.expm(logT)
#             T = SE3.exp(logT)

#             np.testing.assert_allclose(T_true, T.arr)

#         for i in range(100): #Test small thetas
#             u = np.random.uniform(-10, 10, size=3)
#             w = np.random.uniform(-1.0, 1.0, size=3)
#             ang = np.random.uniform(-1e-3, 1e-3)
#             w = w / np.linalg.norm(w) * ang

#             arr = np.concatenate((w, u))

#             T_true = sp.linalg.expm(SE3.hat(arr))
#             T = SE3.Exp(arr)

#             np.testing.assert_allclose(T_true, T.arr)

#     def testVee(self):
#         for i in range(100):
#             u = np.random.uniform(-10, 10, size=3)
#             w = np.random.uniform(-np.pi, np.pi, size=3)

#             arr_true = np.hstack((w,u))

#             logR = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
#             logT = np.zeros((4,4))
#             logT[:3,:3] = logR
#             logT[:3,3] = u

#             arr = SE3.vee(logT)

#             np.testing.assert_allclose(arr_true, arr)

#     def testHat(self):
#         for i in range(100):
#             t = np.random.uniform(-10, 10, size=3)
#             R = Rotation.random().as_matrix()

#             T = SE3.fromRotationMatrix(t, R)
#             logT_true = SE3.log(T)
#             arr = SE3.vee(logT_true)

#             logT = SE3.hat(arr)

#             np.testing.assert_allclose(logT_true, logT)

#     def testAdj(self):
#         for i in range(100):
#             t = np.random.uniform(-10, 10, size=3)
#             R = Rotation.random().as_matrix()
#             T = SE3.fromRotationMatrix(t, R)

#             u = np.random.uniform(-1.0, 1.0, size=3)
#             w = np.random.uniform(-np.pi, np.pi, size=3)
#             delta = np.concatenate((w, u))

#             Adj_T = T.Adj

#             T1_true = T * SE3.Exp(delta)
#             T1 = SE3.Exp(Adj_T @ delta) * T

#             np.testing.assert_allclose(T1_true.arr, T1.arr) #This one is not working

#     def testInv(self):
#         for i in range(100):
#             t = np.random.uniform(-10, 10, size=3)
#             R = Rotation.random().as_matrix()

#             T = SE3.fromRotationMatrix(t, R)

#             T_inv = T.inv()
#             T_inv_true = np.linalg.inv(T.arr)

#             np.testing.assert_allclose(T_inv_true, T_inv.arr)

#     def testGroupAction(self):
#         for i in range(100):
#             t = np.random.uniform(-10, 10, size=3)
#             t2 = np.random.uniform(-10, 10, size=3)
#             R = Rotation.random().as_matrix()
#             R2 = Rotation.random().as_matrix()

#             T1 = SE3.fromRotationMatrix(t, R)
#             T2 = SE3.fromRotationMatrix(t2, R2)

#             T3 = T1 * T2

#             R3 = R @ R2
#             t3 = R @ t2 + t
#             T3_true = SE3.fromRotationMatrix(t3, R3)

#             np.testing.assert_allclose(T3_true.arr, T3.arr)

#     def testTransVector(self):
#         for i in range(100):
#             t = np.random.uniform(-10, 10, size=3)
#             R = Rotation.random().as_matrix()

#             T = SE3.fromRotationMatrix(t, R)

#             pt = np.random.uniform(-5, 5, size=3)

#             rot_pt = T.transa(pt)
#             rot_pt_true = T.R @ pt + T.t

#             np.testing.assert_allclose(rot_pt_true, rot_pt)

#         for i in range(100):
#             t = np.random.uniform(-10, 10, size=3)
#             R = Rotation.random().as_matrix()

#             T = SE3.fromRotationMatrix(t, R)

#             pt = np.random.uniform(-5, 5, size=3)

#             rot_pt = T.transp(pt)
#             T_inv = T.inv()
#             rot_pt_true = T_inv.R @ pt + T_inv.t

#             np.testing.assert_allclose(rot_pt_true, rot_pt)

#     def testFromRPY(self):
#         for i in range(100):
#             t = np.random.uniform(-10, 10, size=3)
#             rpy = np.random.uniform(-np.pi, np.pi, size=3)

#             T = SE3.fromRPY(t, rpy)
#             R_true = Rotation.from_euler('ZYX', [rpy[2], rpy[1], rpy[0]]).as_matrix()
#             T_true = SE3.fromRotationMatrix(t, R_true)

#             np.testing.assert_allclose(T_true.arr, T.arr)

#     def testFromAxisAngle(self):
#         for i in range(100):
#             t = np.random.uniform(-10, 10, size=3)
#             ang = np.random.uniform(-np.pi, np.pi)
#             vec = np.random.uniform(-1, 1, size=3)
#             vec = vec / np.linalg.norm(vec) * ang

#             T = SE3.fromAxisAngle(t, vec)
#             R_true = Rotation.from_rotvec(vec).as_matrix()
#             T_true = SE3.fromRotationMatrix(t, R_true)

#             np.testing.assert_allclose(T_true.arr, T.arr)

#     def testRandom(self):
#         for i in range(100):
#             T = SE3.random()
#             np.testing.assert_allclose(1.0, np.linalg.det(T.R))

#     def testBoxPlus(self):
#         for i in range(100):
#             T = SE3.random()
#             w = np.random.uniform(-np.pi, np.pi, size=3)
#             v = np.random.uniform(-3, 3, size=3)
#             vec = np.array([*w, *v])

#             Tres = T.boxplus(vec)
#             Tres_true = T * SE3.Exp(vec)

#             np.testing.assert_allclose(Tres_true.T, Tres.T)

#     def testBoxMinus(self):
#         for i in range(100):
#             T1 = SE3.random()
#             T2 = SE3.random()

#             w = T1.boxminus(T2)
#             w_true = SE3.Log(T2.inv() * T1)

#             np.testing.assert_allclose(w_true, w)

#     def testNormalize(self):
#         for i in range(10):
#             T = SE3.random()
#             for i in range(10):
#                 T = T * T

#             T.normalize()

#             np.testing.assert_allclose(1, np.linalg.det(T.R))

if __name__=="__main__":
    unittest.main()
