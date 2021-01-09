import numpy as np
from quaternion import Quaternion, skew # move skew to a different file

class SE3:
    def __init__(self, q, t):
        assert isinstance(q, Quaternion)
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

    @classmethod
    def from7vec(cls, arr):
        t = arr[:3]
        q = Quaternion(arr[3:])
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
        arr = np.array([*self.t,*self.q_arr])
        return arr

    @property
    def Adj(self):
        R = self.R
        tx = skew(self.t)
        return np.block([[R, tx @ R], [np.zeros((3,3)), R]])

    def __mul__(self, T):
        q = self.q * T.q
        t = self.t + self.q.rota(T.t)
        return SE3(q,t)

    def __str__(self):
        return str(self.T)

    def __repr__(self):
        q_str = self.q.__repr__()
        t_str = str(self.t)
        return t_str + " " + q_str

    def inv(self, Jr=False, Jl=False):
        q_inv = self.q.inv()
        t_inv = -q_inv.rota(self.t)
        if Jr:
            return SE3(q_inv, t_inv), -self.Adj
        elif Jl:
            T_inv = SE3(q_inv, t_inv)
            return T_inv, -T_inv.Adj
        else:
            return SE3(q_inv, t_inv)

    def transa(self, v, Jr=False, Jl=False):
        vp = self.t + self.q.rota(v)
        if Jr:
            J = np.block([self.R, -self.R @ skew(v)])
            return vp, J
        elif Jl:
            J = np.block([np.eye(3), -skew(self.t) - self.R @ skew(v) @ self.R.T])
            return vp, J
        else:
            return vp

    def transp(self, v):
        T_inv = self.inv()
        return T_inv.transa(v)

    def normalize(self):
        self.q.normalize()

    def boxplusr(self, v):
        return self * SE3.Exp(v)

    def boxminusr(self, T):
        return SE3.Log(T.inv() * self)

    def boxplusl(self, v):
        return SE3.Exp(v) * self

    def boxminusl(self, T):
        return SE3.Log(self * T.inv())

    def compose(self, T, Jr=False, Jl=False):
        res = self * T
        if Jr:
            return res, T.inv().Adj
        if Jl:
            return res, np.eye(6)
        return res

    @staticmethod
    def Identity():
        q = Quaternion.Identity()
        return SE3(q, np.zeros(3))

    @staticmethod
    def log(T, Jr=False, Jl=False):
        if Jr:
            logq, Jq_inv = Quaternion.log(T.q, Jr=Jr)
        elif Jl:
            logq, Jq_inv = Quaternion.log(T.q, Jl=Jl)
        else:
            logq = Quaternion.log(T.q)

        w = logq[1:]
        theta = np.linalg.norm(w)

        wx = skew(w)
        if theta > 1e-3:
            A = np.sin(theta) / theta
            B = (1.0 - np.cos(theta)) / (theta**2)
            V_inv = np.eye(3) - 0.5 * wx + 1/(theta**2) * (1 - A/(2*B)) * (wx @ wx)
            logt = V_inv @ T.t
        else:
            logt = T.t
        logT = np.array([*logt, *logq])

        vx = skew(logt)
        if Jr:
            wx = -wx
            vx = -vx
        if Jl or Jr:
            ct, st = np.cos(theta), np.sin(theta)
            wx2 = wx @ wx
            Q = 0.5 * vx +  (theta - st)/theta**3 * (wx @ vx + vx @ wx + wx @ vx @ wx) - (1 - theta**2/2 - ct)/theta**4 * (wx2 @ vx + vx @ wx2 - 3 * wx @ vx @ wx) - 0.5 * ((1 - theta**2/2 - ct)/theta**4 - 3 * (theta - st - theta**3/6)/theta**5) * (wx @ vx @ wx2 + wx2 @ vx @ wx)
            J = np.block([[Jq_inv, -Jq_inv @ Q @ Jq_inv],
                          [np.zeros((3,3)), Jq_inv]])
            return logT, J
        else:
            return logT

    @staticmethod
    def Log(T, Jr=False, Jl=False):
        if Jr:
            logT, J = SE3.log(T, Jr=Jr)
            return SE3.vee(logT), J
        elif Jl:
            logT, J = SE3.log(T, Jl=Jl)
            return SE3.vee(logT), J
        else:
            logT = SE3.log(T)
            return SE3.vee(logT)

    @classmethod
    def exp(cls, logT, Jr=False, Jl=False):
        v = logT[:3]
        w = logT[4:]
        theta = np.linalg.norm(w)

        if Jr:
            q, Jq = Quaternion.Exp(w, Jr=Jr)
        elif Jl:
            q, Jq = Quaternion.Exp(w, Jl=Jl)
        else:
            q = Quaternion.Exp(w)

        wx = skew(w)
        if theta > 1e-8:
            V = np.eye(3) + (1 - np.cos(theta))/theta**2 * wx + (theta - np.sin(theta)) / theta**3 * (wx @ wx)
            t = V @ v
        else:
            t = v

        vx = skew(v)
        if Jr:
            vx = -vx
            wx = -wx
        if Jl or Jr: # Consider doing a taylor series on Q (should simplify quite a bit)
            ct, st = np.cos(theta), np.sin(theta)
            wx2 = wx @ wx
            Q = 0.5 * vx +  (theta - st)/theta**3 * (wx @ vx + vx @ wx + wx @ vx @ wx) - (1 - theta**2/2 - ct)/theta**4 * (wx2 @ vx + vx @ wx2 - 3 * wx @ vx @ wx) - 0.5 * ((1 - theta**2/2 - ct)/theta**4 - 3 * (theta - st - theta**3/6)/theta**5) * (wx @ vx @ wx2 + wx2 @ vx @ wx)
            J = np.block([[Jq, Q], [np.zeros((3,3)), Jq]])
            return cls(q,t), J
        else:
            return cls(q,t)

    @staticmethod
    def Exp(vec, Jr=False, Jl=False):
        logT = SE3.hat(vec)
        if Jr:
            return SE3.exp(logT, Jr=Jr)
        elif Jl:
            return SE3.exp(logT, Jl=Jl)
        else:
            return SE3.exp(logT)

    @staticmethod
    def hat(vec):
        return np.array([*vec[:3], 0, *vec[3:]])

    @staticmethod
    def vee(vec):
        return np.array([*vec[:3], *vec[4:]])
