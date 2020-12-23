import numpy as np

G = np.array([[[0, 0, 0],
                [0, 0, -1],
                [0, 1, 0]],
               [[0, 0, 1],
                [0, 0, 0],
                [-1, 0, 0]],
               [[0, -1, 0],
                [1, 0, 0],
                [0, 0, 0]]])

class SO3:
    def __init__(self, R):
        assert (R.shape == (3,3))
        self.arr = R

    def __mul__(self, R2):
        assert isinstance(R2, SO3)
        return SO3(self.R @ R2.R)

    def __sub__(self, R2):
        assert isinstance(R2, SO3)
        return self.R - R2.R

    def __str__(self):
        return str(self.R)

    def __repr__(self):
        return str(self.R)

    def inv(self, Jr=False, Jl=False):
        if Jr:
            return SO3(self.arr.T), -self.Adj
        elif Jl:
            R_inv = SO3(self.arr.T)
            return R_inv, -R_inv.Adj
        else:
            return SO3(self.arr.T)

    def transpose(self):
        return SO3(self.arr.T)

    def rota(self, v):
        assert v.size == 3
        return self.arr @ v

    def rotp(self, v):
        assert v.size == 3
        return self.arr.T @ v

    def boxplusr(self, v):
        assert(v.size == 3)
        return self * SO3.Exp(v)

    def boxminusr(self, R2):
        assert isinstance(R2, SO3)
        return SO3.Log(R2.inv() * self)

    def boxplusl(self, v):
        assert(v.size == 3)
        return SO3.Exp(v) * self

    def boxminusl(self, R2):
        assert isinstance(R2, SO3)
        return SO3.Log(self * R2.inv())

    def normalize(self):
        x = self.R[:,0]
        x = x / np.linalg.norm(x)
        y = np.cross(self.R[:,2], x)
        y = y / np.linalg.norm(y)
        z = np.cross(x, y)

        self.arr = np.array([[*x], [*y], [*z]]).T

    def det(self):
        return np.linalg.det(self.R)

    @property
    def R(self):
        return self.arr

    @classmethod
    def fromRPY(cls, angles):
        phi = angles[0]
        theta = angles[1]
        psi = angles[2]

        cps = np.cos(psi)
        sps = np.sin(psi)
        R1 = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])

        ct = np.cos(theta)
        st = np.sin(theta)
        R2 = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])

        cp = np.cos(phi)
        sp = np.sin(phi)
        R3 = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])

        return cls(R1 @ R2 @ R3)

    @classmethod
    def fromAxisAngle(cls, w):
        theta = np.linalg.norm(w)
        skew_w = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

        if np.abs(theta) > 1e-8:
            A = np.sin(theta) / theta
            B = (1 - np.cos(theta)) / (theta**2)
        else:
            A = 1.0 - theta**2 / 6.0 + theta**4 / 120.0
            B = 0.5 - theta**2 / 24.0 + theta**4/720.0

        arr = np.eye(3) + A * skew_w + B * (skew_w @ skew_w)

        return cls(arr)

    @classmethod
    def fromQuaternion(cls, q):
        qw = q[0]
        qv = q[1:]
        qv_skew = np.array([[0, -qv[2], qv[1]], [qv[2], 0, -qv[0]], [-qv[1], qv[0], 0]])

        R = (2 * qw**2 - 1) * np.eye(3) + 2 * qw * qv_skew + 2 * np.outer(qv, qv)
        return cls(R)

    @classmethod
    def random(cls):
        x = np.random.uniform(0, 1, size=3)
        psi = 2 * np.pi * x[0]
        R = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        v = np.array([np.cos(2 * np.pi * x[1]) * np.sqrt(x[2]),
                     np.sin(2 * np.pi * x[1]) * np.sqrt(x[2]),
                     np.sqrt(1 - x[2])])
        H = np.eye(3) - 2 * np.outer(v, v)
        return cls(-H @ R)

    @staticmethod
    def Identity():
        return SO3(np.eye(3))

    @staticmethod
    def log(R): #This function isn't entirely stable but tests pass
        assert isinstance(R, SO3)

        theta = np.arccos((np.trace(R.arr) - 1)/2.0)
        if np.abs(theta) < 1e-8: # Do taylor series expansion
            temp = 1/2.0 * (1 + theta**2 / 6.0 + 7 * theta**4 / 360)
            return temp * (R - R.transpose())
        elif np.abs(np.abs(theta) - np.pi) < 1e-3:
            temp = - np.pi/(theta - np.pi) - 1 - np.pi/6 * (theta - np.pi) - (theta - np.pi)**2/6 - 7*np.pi/360 * (theta - np.pi)**3 - 7/360.0 * (theta - np.pi)**4
            return temp/2.0 * (R - R.transpose())
        else:
            return theta / (2.0 * np.sin(theta)) * (R - R.transpose())

    @classmethod
    def Log(cls, R): #easy call to go straight to a vector
        logR = cls.log(R)
        return cls.vee(logR)

    @classmethod
    def exp(cls, logR):
        assert logR.shape == (3,3)

        w = cls.vee(logR)
        theta = np.sqrt(w @ w)
        if np.abs(theta) > 1e-8:
            R = np.eye(3) + np.sin(theta)/theta * logR + (1 - np.cos(theta))/ (theta**2) * (logR @ logR)
        else: # Do taylor series expansion for small thetas
            R = np.eye(3)

        return cls(R)

    @classmethod
    def Exp(cls, w): # one call to go straight from vector to SO3 object
        logR = cls.hat(w)
        R = cls.exp(logR)
        return R

    @staticmethod
    def vee(logR):
        assert logR.shape == (3,3)
        omega = np.array([logR[2,1], logR[0,2], logR[1,0]])
        return omega

    @staticmethod
    def hat(omega):
        assert omega.size == 3
        return (G @ omega).squeeze()

    @property
    def Adj(self):
        return self.arr

    #Left and right jacobians go here. Read about them first. Seem like a lot of them to implement...
