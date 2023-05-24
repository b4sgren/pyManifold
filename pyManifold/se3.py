import numpy as np

from .quaternion import Quaternion, skew  # move skew to a different file

# from quaternion import Quaternion, skew  # move skew to a different file


class SE3:
    def __init__(self, q, t):
        assert isinstance(q, Quaternion)
        self.q_ = q
        self.t_ = t

    def isValidTransform(self):
        q_norm = self.q_.norm()
        return np.abs(q_norm - 1.0) <= 1e-8

    @classmethod
    def random(cls):
        q = Quaternion.random()
        t = np.random.uniform(-10.0, 10.0, size=3)
        return cls(q, t)

    @classmethod
    def fromRAndt(cls, R, t):
        q = Quaternion.fromRotationMatrix(R)
        return cls(q, t)

    @classmethod
    def fromRPYandt(cls, rpy, t):
        """
        JUST BE AWARE THAT RPY produces qi_from_b but Rb_from_i. Verify what
        frame t is in so that the SE3 object makes sense

        Avoid using this function. Convert RPY to q or R
        Use this instead SE3.fromRAndt(SO3.fromRPY(rpy), t) to get Tb_from_i
        """
        q = Quaternion.fromRPY(rpy)
        return cls(q, t)

    @classmethod
    def fromAxisAngleAndt(cls, v, t):
        q = Quaternion.fromAxisAngle(v)
        return cls(q, t)

    @classmethod
    def from7vec(cls, arr):
        t = arr[:3]
        q = Quaternion(arr[3:])
        return cls(q, t)

    @property
    def w(self):
        return self.q_.w

    @property
    def qv(self):
        return self.q_.qv

    @property
    def x(self):
        return self.q_.x

    @property
    def y(self):
        return self.q_.y

    @property
    def z(self):
        return self.q_.z

    @property
    def q(self):
        return self.q_

    @property
    def q_arr(self):
        return self.q_.q

    @property
    def t(self):
        return self.t_

    @property
    def R(self):
        return self.q.R

    @property
    def T(self):
        # arr = np.array([*self.t,*self.q_arr])
        arr = np.block([[self.R, self.t[:, None]], [np.array([0, 0, 0, 1])]])
        return arr

    @property
    def matrix(self):
        arr = np.block([[self.R, self.t[:, None]], [np.array([0, 0, 0, 1])]])
        return arr

    @property
    def x(self):
        return self.t[0]

    @property
    def y(self):
        return self.t[1]

    @property
    def z(self):
        return self.t[2]

    @property
    def euler(self):
        return self.q_.euler

    @property
    def Adj(self):
        R = self.R
        tx = skew(self.t)
        return np.block([[R, tx @ R], [np.zeros((3, 3)), R]])

    def __mul__(self, T):
        q = self.q * T.q
        t = self.t + self.q.rota(T.t)
        return SE3(q, t)

    def __str__(self):
        return str(self.T)

    def __repr__(self):
        q_str = self.q.__repr__()
        t_str = str(self.t)
        return t_str + " " + q_str

    def inv(self, Jr=None, Jl=None):
        q_inv = self.q.inv()
        t_inv = -q_inv.rota(self.t)
        if Jr is not None:
            return SE3(q_inv, t_inv), -self.Adj @ Jr
        elif Jl is not None:
            T_inv = SE3(q_inv, t_inv)
            return T_inv, -T_inv.Adj @ Jl
        else:
            return SE3(q_inv, t_inv)

    def transa(self, v, Jr=None, Jl=None):
        vp = self.t + self.q.rota(v)
        if Jr is not None:
            J = np.block([self.R, -self.R @ skew(v)])
            return vp, J @ Jr
        elif Jl is not None:
            J = np.block([np.eye(3), -skew(vp)])
            return vp, J @ Jl
        else:
            return vp

    def transp(self, v, Jr=None, Jl=None):
        if Jr is not None:
            T_inv, J = self.inv(Jr=Jr)
            return T_inv.transa(v, Jr=J)
        elif Jl is not None:
            T_inv, J = self.inv(Jl=Jl)
            return T_inv.transa(v, Jl=J)
        else:
            T_inv = self.inv()
            return T_inv.transa(v)

    def normalize(self):
        self.q.normalize()

    def boxplusr(self, v, Jr=None, Jl=None):
        if Jr is not None:
            T, J = SE3.Exp(v, Jr=Jr)
            return self.compose(T, Jr2=J)
        elif Jl is not None:
            T, J = SE3.Exp(v, Jl=Jl)
            return self.compose(T, Jl2=J)
        else:
            return self * SE3.Exp(v)

    def boxminusr(self, T, Jr1=None, Jl1=None, Jr2=None, Jl2=None):
        if Jr1 is not None:
            dT, J = T.inv().compose(self, Jr2=Jr1)
            return SE3.Log(dT, Jr=J)
        elif Jl1 is not None:
            dT, J = T.inv().compose(self, Jl2=Jl1)
            return SE3.Log(dT, Jl=J)
        elif Jr2 is not None:
            T_inv, J = T.inv(Jr=Jr2)
            dT, J = T_inv.compose(self, Jr=J)
            return SE3.Log(dT, Jr=J)
        elif Jl2 is not None:
            T_inv, J = T.inv(Jl=Jl2)
            dT, J = T_inv.compose(self, Jl=J)
            return SE3.Log(dT, Jl=J)
        else:
            return SE3.Log(T.inv() * self)

    def boxplusl(self, v, Jr=None, Jl=None):
        if Jr is not None:
            T, J = SE3.Exp(v, Jr=Jr)
            return T.compose(self, Jr=J)
        elif Jl is not None:
            T, J = SE3.Exp(v, Jl=Jl)
            return T.compose(self, Jl=J)
        else:
            return SE3.Exp(v) * self

    def boxminusl(self, T, Jr1=None, Jl1=None, Jr2=None, Jl2=None):
        if Jr1 is not None:
            diff, J = self.compose(T.inv(), Jr=Jr1)
            return SE3.Log(diff, Jr=J)
        elif Jl1 is not None:
            diff, J = self.compose(T.inv(), Jl=Jl1)
            return SE3.Log(diff, Jl=J)
        elif Jr2 is not None:
            T_inv, J = T.inv(Jr=Jr2)
            diff, J = self.compose(T_inv, Jr2=J)
            return SE3.Log(diff, Jr=J)
        elif Jl2 is not None:
            T_inv, J = T.inv(Jl=Jl2)
            diff, J = self.compose(T_inv, Jl2=J)
            return SE3.Log(diff, Jl=J)
        else:
            return SE3.Log(self * T.inv())

    def compose(self, T, Jr=None, Jl=None, Jr2=None, Jl2=None):
        res = self * T
        if Jr is not None:
            return res, T.inv().Adj @ Jr
        elif Jl is not None:
            return res, np.eye(6) @ Jl
        elif Jr2 is not None:
            return res, np.eye(6) @ Jr2
        elif Jl2 is not None:
            return res, self.Adj @ Jl2
        else:
            return res

    @staticmethod
    def Identity():
        q = Quaternion.Identity()
        return SE3(q, np.zeros(3))

    @staticmethod
    def log(T, Jr=None, Jl=None):
        if Jr is not None:
            logq, Jq_inv = Quaternion.log(T.q, Jr=np.eye(3))
        elif Jl is not None:
            logq, Jq_inv = Quaternion.log(T.q, Jl=np.eye(3))
        else:
            logq = Quaternion.log(T.q)

        w = logq[1:]
        theta = np.linalg.norm(w)

        wx = skew(w)
        if theta > 1e-3:
            A = np.sin(theta) / theta
            B = (1.0 - np.cos(theta)) / (theta**2)
            V_inv = (
                np.eye(3)
                - 0.5 * wx
                + 1 / (theta**2) * (1 - A / (2 * B)) * (wx @ wx)
            )
            logt = V_inv @ T.t
        else:
            logt = T.t
        logT = np.array([*logt, *logq])

        vx = skew(logt)
        if Jr is not None:
            wx = -wx
            vx = -vx
            Jl = Jr
        if Jl is not None or Jr is not None:
            ct, st = np.cos(theta), np.sin(theta)
            wx2 = wx @ wx
            Q = (
                0.5 * vx
                + (theta - st) / theta**3 * (wx @ vx + vx @ wx + wx @ vx @ wx)
                - (1 - theta**2 / 2 - ct)
                / theta**4
                * (wx2 @ vx + vx @ wx2 - 3 * wx @ vx @ wx)
                - 0.5
                * (
                    (1 - theta**2 / 2 - ct) / theta**4
                    - 3 * (theta - st - theta**3 / 6) / theta**5
                )
                * (wx @ vx @ wx2 + wx2 @ vx @ wx)
            )
            J = np.block(
                [[Jq_inv, -Jq_inv @ Q @ Jq_inv], [np.zeros((3, 3)), Jq_inv]]
            )
            return logT, J @ Jl
        else:
            return logT

    @staticmethod
    def Log(T, Jr=None, Jl=None):
        if Jr is not None:
            logT, J = SE3.log(T, Jr=Jr)
            return SE3.vee(logT), J
        elif Jl is not None:
            logT, J = SE3.log(T, Jl=Jl)
            return SE3.vee(logT), J
        else:
            logT = SE3.log(T)
            return SE3.vee(logT)

    @classmethod
    def exp(cls, logT, Jr=None, Jl=None):
        v = logT[:3]
        w = logT[4:]
        theta = np.linalg.norm(w)

        if Jr is not None:
            q, Jq = Quaternion.Exp(w, Jr=np.eye(3))
        elif Jl is not None:
            q, Jq = Quaternion.Exp(w, Jl=np.eye(3))
        else:
            q = Quaternion.Exp(w)

        wx = skew(w)
        if theta > 1e-8:
            V = (
                np.eye(3)
                + (1 - np.cos(theta)) / theta**2 * wx
                + (theta - np.sin(theta)) / theta**3 * (wx @ wx)
            )
            t = V @ v
        else:
            t = v

        vx = skew(v)
        if Jr is not None:
            vx = -vx
            wx = -wx
            Jl = Jr
        if (
            Jl is not None or Jr is not None
        ):  # Consider doing a taylor series on Q (should simplify quite a bit)
            ct, st = np.cos(theta), np.sin(theta)
            wx2 = wx @ wx
            Q = (
                0.5 * vx
                + (theta - st) / theta**3 * (wx @ vx + vx @ wx + wx @ vx @ wx)
                - (1 - theta**2 / 2 - ct)
                / theta**4
                * (wx2 @ vx + vx @ wx2 - 3 * wx @ vx @ wx)
                - 0.5
                * (
                    (1 - theta**2 / 2 - ct) / theta**4
                    - 3 * (theta - st - theta**3 / 6) / theta**5
                )
                * (wx @ vx @ wx2 + wx2 @ vx @ wx)
            )
            J = np.block([[Jq, Q], [np.zeros((3, 3)), Jq]])
            return cls(q, t), J @ Jl
        else:
            return cls(q, t)

    @staticmethod
    def Exp(vec, Jr=None, Jl=None):
        logT = SE3.hat(vec)
        if Jr is not None:
            return SE3.exp(logT, Jr=Jr)
        elif Jl is not None:
            return SE3.exp(logT, Jl=Jl)
        else:
            return SE3.exp(logT)

    @staticmethod
    def hat(vec):
        return np.array([*vec[:3], 0, *vec[3:]])

    @staticmethod
    def vee(vec):
        return np.array([*vec[:3], *vec[4:]])
