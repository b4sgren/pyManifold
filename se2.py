import numpy as np

G = np.array([[[0, 0, 1],
                [0, 0, 0],
                [0, 0, 0]],
               [[0, 0, 0],
                [0, 0, 1],
                [0, 0, 0]],
                [[0, -1, 0],
                [1, 0, 0],
                [0, 0, 0]]])
class SE2:
    def __init__(self, T):
        assert T.shape == (3,3)
        self.arr = T

    def inv(self, Jr=None, Jl=None):
        if not Jr is None:
            return SE2.fromRandt(self.R.T, -self.R.T @ self.t), -self.Adj
        if not Jl is None:
            T_inv = SE2.fromRandt(self.R.T, -self.R.T @ self.t)
            return T_inv, -T_inv.Adj
        else:
            return SE2.fromRandt(self.R.T, -self.R.T @ self.t)

    def __mul__(self, T2):
        assert isinstance(T2, SE2)
        return SE2(self.T @ T2.T)

    def __str__(self):
        return str(self.T)

    def __repr(self):
        return str(self.T)

    def transa(self, v, Jr=None, Jl=None):
        assert v.size == 2
        v = np.array([*v, 1])
        vp = (self.T @ v)[:2]
        if not Jr is None:
            one_x = np.array([[0, -1], [1,0]])
            J = np.block([self.R, (self.R @ one_x @ v[:2])[:,None]])
            return vp, J
        elif not Jl is None:
            one_x = np.array([[0, -1], [1,0]])
            J = np.block([np.eye(2), (one_x @ vp)[:,None]])
            return vp, J
        else:
            return vp

    def transp(self, v):
        assert v.size == 2
        v = np.array([*v, 1])
        vp = self.inv().T @ v
        return vp[:2]

    def boxplusr(self, w):
        assert w.size == 3
        return self * SE2.Exp(w)

    def boxminusr(self, T):
        assert isinstance(T, SE2)
        return SE2.Log(T.inv() * self)

    def boxplusl(self, w):
        assert w.size == 3
        return SE2.Exp(w) * self

    def boxminusl(self, T):
        assert isinstance(T, SE2)
        return SE2.Log(self * T.inv())

    def compose(self, T, Jr=None, Jl=None, Jr2=None, Jl2=None):
        res = self * T
        if not Jr is None:
            return res, T.inv().Adj
        elif not Jl is None:
            return res, np.eye(3)
        elif not Jr2 is None:
            return res, np.eye(3)
        elif not Jl2 is None:
            return res, self.Adj
        else:
            return res

    @property
    def Adj(self):
        J = np.array([[0, -1], [1, 0]])
        adj = np.block([[self.R, (-J @ self.t)[:,None]],
                        [np.zeros((1,2)), 1]])

        return adj

    @property
    def R(self):
        return self.arr[:2,:2]

    @property
    def t(self):
        return self.arr[:2,2]

    @property
    def T(self):
        return self.arr

    @classmethod
    def fromAngleAndt(cls, theta, t):
        ct = np.cos(theta)
        st = np.sin(theta)
        R = np.array([[ct, -st], [st, ct]])

        return cls.fromRandt(R, t)

    @classmethod
    def fromRandt(cls, R, t):
        assert R.shape == (2,2)
        assert t.size == 2
        T = np.block([[R, t[:,None]], [np.zeros(2), 1]])
        return cls(T)

    @staticmethod
    def Identity():
        return SE2(np.eye(3))

    @staticmethod
    def log(T, Jr=None, Jl=None):
        assert isinstance(T, SE2)
        theta = np.arctan2(T.arr[1,0], T.arr[0,0])
        t = T.t

        if np.abs(theta) > 1e-8:
            A = np.sin(theta)/theta
            B = (1 - np.cos(theta))/theta
        else:
            A = 1 - theta**2 / 6.0 + theta**4 / 120.0
            B = theta/2.0 - theta**3 / 24.0 + theta**5/720.0
        normalizer = 1 / (A**2 + B**2)
        V_inv = normalizer * np.array([[A, B], [-B, A]])

        logT = np.zeros((3,3))
        logT[:2,2] = V_inv @ t
        logT[0,1] = -theta
        logT[1,0] = theta

        if not Jr is None:
            p = logT[:2,2]
            u1 = (theta * p[0] - p[1] + p[1] * np.cos(theta) - p[0] * np.sin(theta))/(theta**2)
            u2 = (p[0] + theta * p[1] - p[0] * np.cos(theta) - p[1] * np.sin(theta))/(theta**2)
            den = A**2 + B**2

            w1 = (-B * (-B/A*u1 - u2))/den - u1/A
            w2 = A/den * (-B/A*u1 - u2)
            J = np.array([[A/den, -B/den, w1],
                          [B/den, A/den, w2],
                          [0, 0, 1]])
            return logT, J
        elif not Jl is None:
            p = logT[:2,2]
            u1 = (theta * p[0] + p[1] - p[1] * np.cos(theta) - p[0] * np.sin(theta))/(theta**2)
            u2 = (-p[0] + theta * p[1] + p[0] * np.cos(theta) - p[1] * np.sin(theta))/(theta**2)
            den = A**2 + B**2
            w1 = B/den*(B/A*u1 -u2) - u1/A
            w2 = (B*u1 - A*u2)/den
            J = np.array([[A/den, B/den, w1],
                          [-B/den, A/den, w2],
                          [0, 0, 1]])
            return logT, J
        else:
            return logT

    @classmethod
    def Log(cls, T, Jr=None, Jl=None):
        if not Jr is None:
            logT, J = cls.log(T, Jr=Jr)
            return cls.vee(logT), J
        elif not Jl is None:
            logT, J = cls.log(T, Jl=Jl)
            return cls.vee(logT), J
        else:
            return cls.vee(cls.log(T))

    @classmethod
    def exp(cls, X, Jr=None, Jl=None): #Taylor series expansion
        assert X.shape == (3,3)
        theta = X[1,0]

        if np.abs(theta) > 1e-8:
            A = np.sin(theta)/theta
            B = (1 - np.cos(theta))/theta
        else:
            A = 1 - theta**2 / 6.0 + theta**4 / 120.0
            B = theta/2.0 - theta**3 / 24.0 + theta**5/720.0

        V = np.array([[A, -B], [B, A]])
        t = V @ X[:2,2]
        T = cls.fromAngleAndt(theta, t)

        if not Jr is None:
            p = X[:2, -1]
            u1 = (theta * p[0] - p[1] + p[1] * np.cos(theta) - p[0] * np.sin(theta))/(theta**2)
            u2 = (p[0] + theta * p[1] - p[0] * np.cos(theta) - p[1] * np.sin(theta))/(theta**2)
            J = np.array([[A, B, u1],
                          [-B, A, u2],
                          [0, 0, 1]])
            return T, J
        elif not Jl is None:
            p = X[:2, -1]
            u1 = (theta * p[0] + p[1] - p[1] * np.cos(theta) - p[0] * np.sin(theta))/(theta**2)
            u2 = (-p[0] + theta * p[1] + p[0] * np.cos(theta) - p[1] * np.sin(theta))/(theta**2)
            J = np.array([[A, -B, u1],
                          [B, A, u2],
                          [0, 0, 1]])
            return T, J
        else:
            return T

    @classmethod
    def Exp(cls, vec, Jr=None, Jl=None):
        logR = cls.hat(vec)
        if not Jr is None:
            return cls.exp(logR, Jr=Jr)
        elif not Jl is None:
            return cls.exp(logR, Jl=Jl)
        else:
            return cls.exp(logR)

    @staticmethod
    def vee(X):
        assert X.shape == (3,3)
        arr = np.zeros(3)
        arr[:2] = X[:2,2]
        arr[2] = X[1,0]

        return arr

    @staticmethod
    def hat(arr):
        assert arr.size == 3
        return np.sum(G * arr[:,None, None], axis=0)

    @classmethod
    def random(cls):
        theta = np.random.uniform(-np.pi, np.pi)
        t = np.random.uniform(-5, 5, size=2)
        return cls.fromAngleAndt(theta, t)
