import numpy as np 
import scipy as sp 
from scipy.spatial.transform import Rotation
import unittest 
import sys 
sys.path.append('..')
from quaternion import Quaternion

from IPython.core.debugger import Pdb 

def switchOrder(q):
    temp = q[-1]
    q[1:] = q[:3] # To get rotation matrices to work I need to negate this
    q[0] = temp 

    return q

class Quaternion_Testing(unittest.TestCase):
    def testVee(self):
        for i in range(100):
            q = Rotation.random().as_quat() #Note that the scalar is the last element
            q = switchOrder(q)

            w_true = q[1:]
            w = Quaternion.vee(q)

            np.testing.assert_allclose(w_true, w)

    def testHat(self):
        for i in range(100):
            vec = np.random.uniform(-1.0, 1.0, size=3)

            log_q = Quaternion.hat(vec)
            log_q_true = np.array([0, vec[0], vec[1], vec[2]])

            np.testing.assert_allclose(log_q_true, log_q)
    
    def testFromRot(self): #Not convinced that FromRotationMatrix is actually working
        for i in range(100):
            R = Rotation.random().as_dcm()
            q1 = Quaternion.fromRotationMatrix(R)
            theta = np.arccos((np.trace(R) - 1)/2) 
            v = -np.array([R[2,1]-R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / (2 * np.sin(theta))
            q_true = np.zeros(4)
            q_true[0] = np.cos(theta/2)
            q_true[1:] = np.sin(theta/2) * v
 
            if np.sign(q_true[0]) < 0.0:
                q_true *= -1
 
            if np.linalg.norm(q_true - q1.arr) > 1e-3:
                Pdb().set_trace()
                q2 = Quaternion.fromRotationMatrix(R)
 
            np.testing.assert_allclose(q_true, q1.arr)
    
    def testFromAxisAngle(self):
        for i in range(100):
            vec = Rotation.random().as_rotvec()
            q1 = Quaternion.fromAxisAngle(vec)
            q_true = Rotation.from_rotvec(vec).as_quat()
            q_true = switchOrder(q_true)

            if np.linalg.norm(q_true - q1.arr) > 1e-3:
                Pdb().set_trace()
                q2 = Quaternion.fromAxisAngle(vec)

            np.testing.assert_allclose(q_true, q1.arr)
    
    def testFromRPY(self):
        for i in range(100):
            ang = np.random.uniform(-np.pi, np.pi, size=3)
            q1 = Quaternion.fromRPY(ang[0], ang[1], ang[2])
            q_true = Rotation.from_euler('ZYX', [ang[2], ang[1], ang[0]]).as_quat()
            q_true = switchOrder(q_true)

            if q_true[0] < 0.0:
                q_true *= -1
            
            if np.linalg.norm(q_true - q1.arr) > 1e-3:
                Pdb().set_trace()
                q2 = Quaternion.fromRPY(ang[0], ang[1], ang[2])
            
            np.testing.assert_allclose(q_true, q1.arr)
    
    def testQuaternionMultiply(self):
        for i in range(100):
            rot1 = Rotation.random()
            rot2 = Rotation.random()

            q_true = (rot1 * rot2).as_quat()
            q_true = switchOrder(q_true)
            if q_true[0] < 0.0:
                q_true *= -1

            q1 = Quaternion(switchOrder(rot1.as_quat()))
            q2 = Quaternion(switchOrder(rot2.as_quat()))
            q3 = q1 * q2

            if np.linalg.norm(q_true - q3.arr) > 1e-3:
                Pdb().set_trace()
                q4 = q1 * q2

            np.testing.assert_allclose(q_true, q3.arr)

    def testInv(self):
        for i in range(100):
            R = Rotation.random().as_quat()
            R = switchOrder(R)

            q = Quaternion(R)
            q_inv = q.inv()
            res = q * q_inv 
            res_true = np.array([1.0, 0.0, 0.0, 0.0])

            np.testing.assert_allclose(res_true, res.arr, atol=1e-8)
    
    def testLog(self): 
        for i in range(100):
            r = Rotation.random()
            q = Quaternion(switchOrder(r.as_quat()))
            R = r.as_dcm()

            w = Quaternion.log(q)[1:]
            logR = sp.linalg.logm(R)
            w_true = np.array([-logR[1,2], logR[0,2], -logR[0,1]])

            np.testing.assert_allclose(w_true, w)
        
        for i in range(100): #Taylor series
            ang = np.random.uniform(-1e-3, 1e-3)
            vec = np.random.uniform(-1, 1, size=3)
            w_true = vec / np.linalg.norm(vec) * ang

            q = Quaternion.fromAxisAngle(w_true)
            w1 = Quaternion.Log(q)
            if np.linalg.norm(w_true - w1) > 1e-3:
                Pdb().set_trace()
                w2 = Quaternion.Log(q)
            
            np.testing.assert_allclose(w_true, w1)

    
    def testExp(self): 
        for i in range(100):
            r = Rotation.random()
            R = r.as_dcm()
            logR = sp.linalg.logm(R)
            w = np.array([0, -logR[1,2], logR[0,2], -logR[0,1]])

            q = Quaternion.exp(w)
            q_true = switchOrder(r.as_quat())
            if q_true[0] < 0.0:
                q_true *= -1

            if np.linalg.norm(q_true - q.arr) > 1e-3:
                Pdb().set_trace()
                q2 = Quaternion.exp(w)

            np.testing.assert_allclose(q_true, q.arr)
        
        for i in range(100): #Taylor series portion
            ang = np.random.uniform(-1e-3, 1e-3)
            vec = np.random.uniform(-1, 1, size=3)
            w = vec / np.linalg.norm(vec) * ang

            q = Quaternion.Exp(w)
            logR = Quaternion.skew(w)
            R = sp.linalg.expm(logR)
            q_true = switchOrder(Rotation.from_dcm(R).as_quat())
            if np.linalg.norm(q_true - q.arr) > 1e-3:
                Pdb().set_trace()
                q2 = Quaternion.Exp(w)

            np.testing.assert_allclose(q_true, q.arr, atol=1e-6)
    
    def testVectorRotation(self): #This isn't working at all
        for i in range(100):
            v = np.random.uniform(-10, 10, size=3)
            r = Rotation.random()
            q1 = switchOrder(r.as_quat())
            my_q = Quaternion(q1)
            R = r.as_dcm().T

            vp = R @ v
            my_vp = my_q.rot(v)

            if np.linalg.norm(vp - my_vp) > 1e-3:
                Pdb().set_trace()
                vp2 = my_q.rot(v)

            np.testing.assert_allclose(vp, my_vp)
    
    def testAdjoint(self):
        for i in range(100):
            delta = np.random.uniform(-np.pi, np.pi, size=3)
            rot = Rotation.random()
            q = Quaternion(switchOrder(rot.as_quat()))

            q_true = q * Quaternion.Exp(delta)
            q1 = Quaternion.Exp(q.Adj.rot(delta)) * q
            if np.linalg.norm(q_true.arr - q1.arr) > 1e-3:
                Pdb().set_trace()
                q2 = Quaternion(q.Adj @ delta) * q

            np.testing.assert_allclose(q_true.arr, q1.arr)
        
        # for i in range(100): #Matrix form of adjoint
        #     delta = np.random.uniform(-np.pi, np.pi, size=3)
        #     rot = Rotation.random()
        #     q = Quaternion(switchOrder(rot.as_quat()))

        #     q_true = q * Quaternion.Exp(delta)
        #     q1 = Quaternion.Exp(q.MatAdj @ delta) * q
        #     if np.linalg.norm(q_true.arr - q1.arr) > 1e-3:
        #         Pdb().set_trace()
        #         q2 = Quaternion(q.MatAdj @ delta) * q
            
        #     np.testing.assert_allclose(q_true.arr, q1.arr)

if __name__=="__main__":
    unittest.main()