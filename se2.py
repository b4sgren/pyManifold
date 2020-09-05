import numpy as np

G = np.array([[[0, -1, 0],
                [1, 0, 0],
                [0, 0, 0]],
               [[0, 0, 1],
                [0, 0, 0],
                [0, 0, 0]],
               [[0, 0, 0],
                [0, 0, 1],
                [0, 0, 0]]])
class SE2:
    def __init__(self, T):
        assert T.shape == (3,3)
        self.arr = T

    def inv(self):
        return SE2.fromRandt(self.R.T, -self.R.T @ self.t)

    def __mul__(self, T2):
        assert isinstance(T2, SE2)
        return SE2(self.T @ T2.T)

    def __str__(self):
        return str(self.T)

    def __repr(self):
        return str(self.T)

    def transa(self, v):
        assert v.size == 2
        v = np.array([*v, 1])
        vp = self.T @ v
        return vp[:2]

    def transp(self, v):
        assert v.size == 2
        v = np.array([*v, 1])
        vp = self.inv().T @ v
        return vp[:2]

    def boxplus(self, w):
        assert w.size == 3
        return self * SE2.Exp(w)

    def boxminus(self, T):
        assert isinstance(T, SE2)
        return SE2.Log(T.inv() * self)

    @property
    def Adj(self):
        J = np.array([[0, 1], [-1, 0]])
        adj = np.zeros((3,3))
        adj[0,0] = 1
        adj[1:,0] = J @ self.t
        adj[1:,1:] = self.R

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
    def log(T):
        assert isinstance(T, SE2)
        theta = np.arctan2(T.arr[1,0], T.arr[0,0])
        t = T.t

        if np.abs(theta) > 1e-3:
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

        return logT

    @classmethod
    def Log(cls, T):
        logT = cls.log(T)
        return cls.vee(logT)

    @classmethod
    def exp(cls, X): #Taylor series expansion
        assert X.shape == (3,3)
        theta = X[1,0]

        if np.abs(theta) > 1e-3:
            A = np.sin(theta)/theta
            B = (1 - np.cos(theta))/theta
        else:
            A = 1 - theta**2 / 6.0 + theta**4 / 120.0
            B = theta/2.0 - theta**3 / 24.0 + theta**5/720.0

        V = np.array([[A, -B], [B, A]])
        t = V @ X[:2,2]

        return cls.fromAngleAndt(theta, t)

    @classmethod
    def Exp(cls, vec):
        logR = cls.hat(vec)
        return cls.exp(logR)

    @staticmethod
    def vee(X):
        assert X.shape == (3,3)
        arr = np.zeros(3)
        arr[1:] = X[:2,2]
        arr[0] = X[1,0]

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
