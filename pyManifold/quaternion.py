import numpy as np

# Get rid of this eventually
from scipy.spatial.transform import Rotation as Rot


def skew(qv):
    return np.array(
        [[0, -qv[2], qv[1]], [qv[2], 0, -qv[0]], [-qv[1], qv[0], 0]]
    )


"""
Quaternion Convention:
This class has been developed to follow the Hamiltonian convention for
quaternions. In other words the quaternion looks like qw + qxi + qyj + qzk
and ij=k, jk=i, ki=j, i^2=j^2=k^2=ijk=-1. However, one of the disadvantages
of the original hamiltonian quaternion is that the rotation matrices formed
from the quaternion multiply in the opposite order of the quaternions themselves
R(q1)R(q2) = R(q2*q1). This is a result of (what I consider to be) a poor
choice of conversion between quaternions and rotation matrices. I have modified
the convention such that R(q1)R(q2)=R(q1*q2) to keep the behavior of
quaternions consistent with rotation matrices. Other quaternion conventions
exist, most predominantly the JPL/Shuster convention. The existence of different
conventions can be confusing and the following references helped me sort out
the confusing aspects.

References:
Blog post: https://fzheng.me/2017/11/12/quaternion_conventions_en/
Blog post: https://possiblywrong.wordpress.com/2021/05/10/beware-the-natural-quaternion/
Hamiltonian Quaternions: file:///home/brendon/Downloads/giq-1-2000-127-143.pdf : Quaternions and Rotation Sequences by Jack Kuipers
JPL Quaternions: http://malcolmdshuster.com/Pubp_021_072x_J_JAS0000_quat_MDS.pdf
Quaternion Kinematics for the Err State Kalman Filter: https://arxiv.org/pdf/1711.02508.pdf
Why and how to avoid the flipped quaternion: https://arxiv.org/pdf/1801.07478.pdf



The way this class is written it behaves just like a rotation matrix and can
serve as a drop in replacement for the SO3 class.
"""


class Quaternion:
    def __init__(self, q):
        if isinstance(q, np.ndarray):
            if q.shape == (4,) or q.shape == (4, 1) or q.shape == (1, 4):
                self.arr = q.squeeze()
                if self.arr[0] < 0:
                    self.arr *= -1
            else:
                raise ValueError("Input must be a numpy array of length 4")
        else:
            # Initialize to identity
            self.arr = np.array([1, 0, 0, 0])

    @property
    def w(self):
        return self.arr[0]

    @property
    def x(self):
        return self.arr[1]

    @property
    def y(self):
        return self.arr[2]

    @property
    def z(self):
        return self.arr[3]

    @property
    def qv(self):
        return self.arr[1:]

    @property
    def q(self):
        return self.arr

    @property
    def R(self):
        return (
            (2 * self.w**2 - 1) * np.eye(3)
            + 2 * self.w * skew(self.qv)
            + 2 * np.outer(self.qv, self.qv)
        )

    @property
    def euler(self) -> np.ndarray:
        w, x, y, z = self.w, self.x, self.y, self.z

        phi = np.arctan2(2 * (w * x + y * z), w**2 + z**2 - x**2 - y**2)
        theta = np.arcsin(2 * (w * y - x * z))
        psi = np.arctan2(2 * (w * z + x * y), w**2 + x**2 - y**2 - z**2)

        return np.array([phi, theta, psi])

    @property
    def Adj(self):
        return self.R

    def __mul__(self, q):
        return self.otimes(q)

    def __str__(self):
        return str(self.q)

    def __repr__(self):
        return f"[{self.w} + {self.x}i + {self.y}j + {self.z}k]"

    def otimes(self, q):
        # Ql in Quat Kinematics for Err St Kalman Filter
        Q = np.block(
            [
                [self.w, -self.qv],
                [self.qv[:, None], self.w * np.eye(3) + self.skew()],
            ]
        )
        return Quaternion(Q @ q.q)

    def skew(self):
        qv = self.qv
        return skew(qv)

    def inv(self, Jr=None, Jl=None):
        if not Jr is None:
            return Quaternion(np.array([self.w, *(-self.qv)])), -self.Adj @ Jr
        elif not Jl is None:
            q_inv = Quaternion(np.array([self.w, *(-self.qv)]))
            return q_inv, -q_inv.Adj @ Jl
        return Quaternion(np.array([self.w, -self.x, -self.y, -self.z]))

    # Does q*v*q.inv()
    def rota(self, v, Jr=None, Jl=None):
        qw = self.w
        qv = self.qv

        t = 2 * skew(v) @ qv
        vp = v - qw * t + skew(t) @ qv
        if not Jr is None:
            J = -self.R @ skew(v)
            return vp, J @ Jr
        elif not Jl is None:
            J = -skew(vp)
            return vp, J @ Jl
        else:
            return vp

    def rotp(self, v, Jr=None, Jl=None):
        if not Jr is None:
            q_inv, J = self.inv(Jr=Jr)
            vp, J = q_inv.rota(v, Jr=J)
            return vp, J
        elif not Jl is None:
            q_inv, J = self.inv(Jl=Jl)
            vp, J = q_inv.rota(v, Jl=J)
            return vp, J
        else:
            return self.inv().rota(v)

    def normalize(self):
        self.arr = self.q / self.norm()

    def norm(self):
        return np.linalg.norm(self.q)

    def boxplusr(self, w, Jr=None, Jl=None):
        assert w.size == 3
        if not Jr is None:
            q, J = Quaternion.Exp(w, Jr=Jr)
            return self.compose(q, Jr2=J)
        elif not Jl is None:
            q, J = Quaternion.Exp(w, Jl=Jl)
            return self.compose(q, Jl2=J)
        else:
            return self * Quaternion.Exp(w)

    def boxminusr(self, q, Jr1=None, Jl1=None, Jr2=None, Jl2=None):
        assert isinstance(q, Quaternion)
        if Jr1 is not None:
            dq, J = q.inv().compose(self, Jr2=Jr1)
            return Quaternion.Log(dq, Jr=J)
        elif Jl1 is not None:
            dq, J = q.inv().compose(self, Jl2=Jl1)
            return Quaternion.Log(dq, Jl=J)
        elif Jr2 is not None:
            q_inv, J = q.inv(Jr=Jr2)
            dq, J = q_inv.compose(self, Jr=J)
            return Quaternion.Log(dq, Jr=J)
        elif Jl2 is not None:
            q_inv, J = q.inv(Jl=Jl2)
            dq, J = q_inv.compose(self, Jl=J)
            return Quaternion.Log(dq, Jl=J)
        else:
            return Quaternion.Log(q.inv() * self)

    def boxplusl(self, w, Jr=None, Jl=None):
        assert w.size == 3
        if Jr is not None:
            q, J = Quaternion.Exp(w, Jr=Jr)
            return q.compose(self, Jr=J)
        elif Jl is not None:
            q, J = Quaternion.Exp(w, Jl=Jl)
            return q.compose(self, Jl=J)
        else:
            return Quaternion.Exp(w) * self

    def boxminusl(self, q, Jr1=None, Jl1=None, Jr2=None, Jl2=None):
        assert isinstance(q, Quaternion)
        if Jr1 is not None:
            diff, J = self.compose(q.inv(), Jr=Jr1)
            return Quaternion.Log(diff, Jr=J)
        elif Jl1 is not None:
            diff, J = self.compose(q.inv(), Jl=Jl1)
            return Quaternion.Log(diff, Jl=J)
        elif Jr2 is not None:
            q_inv, J = q.inv(Jr=Jr2)
            diff, J = self.compose(q_inv, Jr2=J)
            return Quaternion.Log(diff, Jr=J)
        elif Jl2 is not None:
            q_inv, J = q.inv(Jl=Jl2)
            diff, J = self.compose(q_inv, Jl2=J)
            return Quaternion.Log(diff, Jl=J)
        else:
            return Quaternion.Log(self * q.inv())

    def compose(self, q, Jr=None, Jl=None, Jr2=None, Jl2=None):
        res = self * q
        if not Jr is None:
            return res, q.inv().Adj @ Jr
        elif not Jl is None:
            return res, np.eye(3) @ Jl
        elif not Jr2 is None:
            return res, np.eye(3) @ Jr2
        elif not Jl2 is None:
            return res, self.Adj @ Jl2
        else:
            return res

    @classmethod
    def random(cls):
        # Method found at planning.cs.uiuc.edu/node198.html (SO how to generate
        # a random quaternion quickly)
        u = np.random.uniform(0.0, 1.0, size=3)
        qw = np.sin(2 * np.pi * u[1]) * np.sqrt(1 - u[0])
        q1 = np.cos(2 * np.pi * u[1]) * np.sqrt(1 - u[0])
        q2 = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
        q3 = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
        return Quaternion(np.array([qw, q1, q2, q3]))

    @classmethod
    def fromRotationMatrix(cls, R):
        d = np.trace(R)
        if d > 0:
            s = 2 * np.sqrt(d + 1)
            q = np.array(
                [
                    s / 4,
                    1 / s * (R[1, 2] - R[2, 1]),
                    1 / s * (R[2, 0] - R[0, 2]),
                    1 / s * (R[0, 1] - R[1, 0]),
                ]
            )
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            q = np.array(
                [
                    1 / s * (R[1, 2] - R[2, 1]),
                    s / 4,
                    1 / s * (R[1, 0] + R[0, 1]),
                    1 / s * (R[2, 0] + R[0, 2]),
                ]
            )
        elif R[1, 1] > R[2, 2]:
            s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            q = np.array(
                [
                    1 / s * (R[2, 0] - R[0, 2]),
                    1 / s * (R[1, 0] + R[0, 1]),
                    s / 4,
                    1 / s * (R[2, 1] + R[1, 2]),
                ]
            )
        else:
            s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            q = np.array(
                [
                    1 / s * (R[0, 1] - R[1, 0]),
                    1 / s * (R[2, 0] + R[0, 2]),
                    1 / s * (R[2, 1] + R[1, 2]),
                    s / 4,
                ]
            )

        q[1:] *= -1

        return Quaternion(q)

    @classmethod
    def fromRPY(cls, rpy):
        """
        Note that the SO3 class produces R^b_i but this class produces the
        inverse or q^i_b. Need to look into this a little bit more
        """
        cp, ct, cps = np.cos(rpy / 2)
        sp, st, sps = np.sin(rpy / 2)

        q = np.zeros(4)
        q[0] = cp * ct * cps + sp * st * sps
        q[1] = sp * ct * cps - cp * st * sps
        q[2] = cp * st * cps + sp * ct * sps
        q[3] = cp * ct * sps - sp * st * cps
        q = q / np.linalg.norm(q)

        return cls(q)

    @classmethod
    def fromAxisAngle(cls, vec):
        return cls.Exp(vec)

    @staticmethod
    def Identity():
        return Quaternion(np.array([1.0, 0.0, 0.0, 0.0]))

    @staticmethod
    def hat(w):
        return np.array([0, *w])

    @staticmethod
    def vee(W):
        return W[1:]

    @staticmethod
    def log(q, Jr=None, Jl=None):
        qw = q.w
        qv = q.qv
        theta = np.linalg.norm(qv)

        if np.abs(theta) > 1e-8:
            w = 2 * np.arctan(theta / qw) * qv / theta
        else:
            temp = (
                1 / qw - theta**2 / (3 * qw**3) + theta**4 / (5 * qw**5)
            )
            w = 2 * temp * qv
        logq = np.array([0, *w])

        if not Jr is None:
            wx = skew(w)
            phi = np.linalg.norm(w)
            J = (
                np.eye(3)
                + 0.5 * wx
                + (1 / phi**2 - (1 + np.cos(phi)) / (2 * phi * np.sin(phi)))
                * (wx @ wx)
            )
            return logq, J @ Jr
        elif not Jl is None:
            wx = skew(w)
            phi = np.linalg.norm(w)
            J = (
                np.eye(3)
                - 0.5 * wx
                + (1 / phi**2 - (1 + np.cos(phi)) / (2 * phi * np.sin(phi)))
                * (wx @ wx)
            )
            return logq, J @ Jl
        else:
            return logq

    @staticmethod
    def Log(q, Jr=None, Jl=None):
        if not Jr is None:
            W, J = Quaternion.log(q, Jr=Jr)
            return Quaternion.vee(W), J
        elif not Jl is None:
            W, J = Quaternion.log(q, Jl=Jl)
            return Quaternion.vee(W), J
        else:
            W = Quaternion.log(q)
            return Quaternion.vee(W)

    @classmethod
    def exp(cls, W, Jr=None, Jl=None):
        vec = W[1:]
        theta = np.linalg.norm(vec)
        if abs(theta) < 1e-5:
            if Jr is not None:
                return cls.Identity(), Jr
            elif Jl is not None:
                return cls.Identity(), Jl
            else:
                return cls.Identity()

        v = vec / theta

        if np.abs(theta) > 1e-8:
            qw = np.cos(theta / 2)
            qv = v * np.sin(theta / 2)
        else:
            qw = 1 - theta**2 / 8 + theta**4 / 46080
            temp = 1 / 2 - theta**2 / 48 + theta**4 / 3840
            qv = vec * temp
        q = cls(np.array([qw, *qv]))

        if not Jr is None:
            thetax = skew(vec)
            J = (
                np.eye(3)
                - (1 - np.cos(theta)) / theta**2 * thetax
                + (theta - np.sin(theta)) / theta**3 * (thetax @ thetax)
            )
            return q, J @ Jr
        elif not Jl is None:
            thetax = skew(vec)
            J = (
                np.eye(3)
                + (1 - np.cos(theta)) / theta**2 * thetax
                + (theta - np.sin(theta)) / theta**3 * (thetax @ thetax)
            )
            return q, J @ Jl
        else:
            return q

    @staticmethod
    def Exp(w, Jr=None, Jl=None):
        W = Quaternion.hat(w)
        if not Jr is None:
            return Quaternion.exp(W, Jr=Jr)
        elif not Jl is None:
            return Quaternion.exp(W, Jl=Jl)
        else:
            return Quaternion.exp(W)


def from_euler(rpy):
    phi, theta, psi = rpy / 2

    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)

    q = np.zeros(4)
    q[0] = cp * ct * cps - sp * st * sps
    q[1] = sp * ct * cps + cp * st * sps
    q[2] = cp * st * cps - sp * ct * sps
    q[3] = cp * ct * sps + sp * st * cps
    q = q / np.linalg.norm(q)

    return Quaternion(q)
