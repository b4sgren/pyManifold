import numpy as np
from quaternion import Quaternion

#Jacobians, and overload plus and minus for box plus and box minus

# Revise so that the quaternion is the quaternion object
class SE3:
    def __init__(self, q, t):
        self.q_ = q
        self.t_ = t

    def isValidTransform(self):
        q_norm = self.q_.norm()
        return (np.abs(q_norm - 1.0) <= 1e-8)

    @classmethod
    def random(cls):
        q = Quaternion.random()
        t = np.random.uniform(-10.0, 10.0, size=3)
        return cls(q,t)

    @property
    def qw(self):
        return self.q_.qw

    @property
    def qv(self):
        return self.q_.qv

    @property
    def qx(self):
        return self.q_.qx

    @property
    def qy(self):
        return self.q_.qy

    @property
    def qz(self):
        return self.q_.qz

    @property
    def q(self):
        return self.q_

    @property
    def q_arr(self):
        return self.q.q

    @property
    def t(self):
        return self.t_

    @property
    def R(self):
        return self.q.R

    @property
    def T(self):
        arr = np.concatenate(self.t,self.q.q)
        return arr

    def __mul__(self, T):
        q = self.q * T.q
        t = self.t + self.q.rota(T.t)
        return SE3(q,t)

    def inv(self):
        q_inv = self.q.inv()
        t_inv = -q_inv.rota(self.t)
        return SE3(q_inv, t_inv)

    def transa(self, v):
        return self.t + self.q.rota(v)

    def transp(self, v):
        T_inv = self.inv()
        return T_inv.transa(v)

    @classmethod
    def fromRAndt(cls, R, t):
        q = Quaternion.fromRotationMatrix(R)
        return cls(q,t)

    @classmethod
    def fromRPYandt(cls, rpy, t):
        q = Quaternion.fromRPY(rpy)
        return cls(q,t)

    @classmethod
    def fromAxisAngleAndt(cls, v, t):
        q = Quaternion.fromAxisAngle(v)
        return cls(q,t)

# class SE3:
#     def __init__(self, T):
#         assert T.shape == (4,4)
#         self.arr = T

#     def inv(self):
#         R_inv = self.R.T
#         t_inv = -R_inv @ self.t
#         return SE3.fromRotationMatrix(t_inv, R_inv)

#     def __mul__(self, T):
#         assert isinstance(T, SE3)
#         temp = self.arr @ T.arr
#         return SE3(temp)

#     def __str__(self):
#         return str(self.T)

#     def __repr__(self):
#         return str(self.T)

#     def transa(self, vec):
#         assert vec.size == 3
#         v = np.array([*vec, 1])
#         vp = self.T @ v
#         return vp[:3]

#     def transp(self, vec):
#         assert vec.size == 3
#         v = np.array([*vec, 1])
#         vp = self.inv().T @ v
#         return vp[:3]

#     def boxplus(self, arr):
#         assert arr.size == 6
#         return self * SE3.Exp(arr)

#     def boxminus(self, T):
#         assert isinstance(T, SE3)
#         return SE3.Log(T.inv() * self)

#     def normalize(self):
#         R = self.R
#         x = R[:,0] / np.linalg.norm(R[:,0])
#         y = np.cross(R[:,2], x)
#         y /= np.linalg.norm(y)
#         z = np.cross(x, y)

#         self.arr[:3, :3] = np.array([[*x], [*y], [*z]]).T

#     @property
#     def R(self):
#         return self.arr[:3,:3]

#     @property
#     def t(self):
#         return self.arr[:3,3]

#     @property
#     def T(self):
#         return self.arr

#     @classmethod
#     def fromRotationMatrix(cls, t, R):
#         assert t.size == 3
#         assert R.shape == (3,3)
#         T = np.block([[R, t[:,None]], [np.zeros(3), 1]])
#         return cls(T)

#     @classmethod
#     def fromRPY(cls, t, angles):
#         assert angles.size == 3
#         assert t.size == 3

#         phi = angles[0]
#         theta = angles[1]
#         psi = angles[2]

#         cps = np.cos(psi)
#         sps = np.sin(psi)
#         R1 = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])

#         ct = np.cos(theta)
#         st = np.sin(theta)
#         R2 = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])

#         cp = np.cos(phi)
#         sp = np.sin(phi)
#         R3 = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])

#         return cls.fromRotationMatrix(t, R1 @ R2 @ R3)

#     @classmethod
#     def fromAxisAngle(cls, t, w):
#         assert t.size == 3
#         assert w.size == 3

#         theta = np.linalg.norm(w)
#         skew_w = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

#         if theta > 1e-3:
#             A = np.sin(theta)/ theta
#             B = (1 - np.cos(theta)) / (theta**2)
#         else:
#             A = 1 - theta**2 / 6.0 + theta**4 / 120.0
#             B = 0.5 - theta**2 / 24.0 + theta**4 / 720.0

#         arr = np.eye(3) + A * skew_w + B * (skew_w @ skew_w)
#         return cls.fromRotationMatrix(t, arr)

#     @staticmethod
#     def log(T):
#         assert isinstance(T, SE3)

#         theta = np.arccos((np.trace(T.R) - 1)/2.0)
#         if np.abs(theta) < 1e-3:
#             temp = 1/2.0 * (1 + theta**2 / 6.0 + 7 * theta**4 / 360)
#             logR =  temp * (T.R - T.R.T)
#         elif np.abs(np.abs(theta) - np.pi) < 1e-3:
#             temp = - np.pi/(theta - np.pi) - 1 - np.pi/6 * (theta - np.pi) - (theta - np.pi)**2/6 - 7*np.pi/360 * (theta - np.pi)**3 - 7/360.0 * (theta - np.pi)**4
#             logR =  temp/2.0 * (T.R - T.R.T)
#         else:
#             logR = theta / (2.0 * np.sin(theta)) * (T.R - T.R.T)

#         if np.abs(theta) > 1e-3:
#             temp = np.sin(theta) / (theta * (1 - np.cos(theta)))
#         else:
#             temp = 2/theta**2 - 1/6.0 - theta**2/360.0 - theta**4/15120.0

#         V_inv = np.eye(3) - 0.5 * logR + (1/theta**2 - temp/2) * (logR @ logR)
#         u = V_inv @ T.t

#         logT = np.zeros((4,4))
#         logT[:3,:3] = logR
#         logT[:3,3] = u

#         return logT

#     @classmethod #One call to go from Transformation matrix to vector
#     def Log(cls, T):
#         logT = cls.log(T)
#         return cls.vee(logT)

#     @classmethod
#     def exp(cls, logT):
#         assert logT.shape == (4,4)

#         u = logT[:3,3]
#         w = np.array([logT[2,1], logT[0,2], logT[1,0]])

#         theta = np.sqrt(w @ w)
#         if np.abs(theta) > 1e-3:
#             A = np.sin(theta) / theta
#             B = (1 - np.cos(theta)) / (theta**2)
#             C = (1 - A) / (theta**2)
#         else: #Taylor series expansion
#             A = 1.0 - theta**2/6.0 + theta**4/120.0
#             B = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
#             C = 1/6.0 - theta**2/120.0 + theta**4 / 5040.0

#         R = np.eye(3) + A * logT[:3,:3] + B * np.linalg.matrix_power(logT[:3,:3], 2)
#         V = np.eye(3) + B * logT[:3,:3] + C * np.linalg.matrix_power(logT[:3,:3], 2)

#         t = V @ u
#         return cls.fromRotationMatrix(t, R)

#     @classmethod
#     def Exp(cls, arr): #One call to go from a vector to Transformation matrix
#         logT = cls.hat(arr)
#         return cls.exp(logT)

#     @classmethod
#     def random(cls):
#         x = np.random.uniform(0, 1, size=3)
#         t = np.random.uniform(-10, 10, size=3)
#         psi = 2 * np.pi * x[0]
#         R = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
#         v = np.array([np.cos(2 * np.pi * x[1]) * np.sqrt(x[2]),
#                      np.sin(2 * np.pi * x[1]) * np.sqrt(x[2]),
#                      np.sqrt(1 - x[2])])
#         H = np.eye(3) - 2 * np.outer(v, v)
#         return cls.fromRotationMatrix(t, -H @ R)

#     @staticmethod
#     def Identity():
#         return SE3(np.eye(4))

#     @staticmethod
#     def vee(logT):
#         assert logT.shape == (4,4)
#         u = logT[:3,3]
#         w = np.array([logT[2,1], logT[0,2], logT[1,0]])

#         return np.concatenate((w, u))

#     @staticmethod
#     def hat(arr):
#         assert arr.size == 6
#         return np.sum(G * arr[:,None, None], axis=0)

#     @property
#     def Adj(self):
#         R = self.R
#         t = self.t

#         adj = np.zeros((6,6))
#         adj[:3,:3] = R
#         adj[-3:,-3:] = R

#         tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
#         adj[3:,:3] = tx @ R

#         return adj

#     #Left and right jacobian stuff goes here
