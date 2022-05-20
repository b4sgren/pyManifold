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

def skew(qv: np.ndarray) -> np.ndarray:
    return np.array([[0, -qv[2], qv[1]], [qv[2], 0, -qv[0]], [-qv[1], qv[0], 0]])

class SO3:
    def __init__(self, R: np.ndarray) -> None:
        """
        Constructor for SO3

        Args:
        R -- Numpy array expressing a rotation matrix
        """
        assert (R.shape == (3,3))
        self.arr = R

    def __mul__(self, R2: 'SO3') -> 'SO3':
        """
        Overloaded * operator

        Args:
        R2 -- An instance of SO3

        Returns:
        The new rotation consisting of self * R2
        """
        assert isinstance(R2, SO3)
        return SO3(self.R @ R2.R)

    # What to do about this. I don't want this public
    def __sub__(self, R2):
        assert isinstance(R2, SO3)
        return self.R - R2.R

    def __str__(self):
        return str(self.R)

    def __repr__(self):
        return str(self.R)

    def inv(self, Jr: np.ndarray = None, Jl: np.ndarray = None) -> 'SO3':
        """
        Computes the inverse of the SO3 instance

        Keyword Args:
        Jr -- If specified it computes the right jacobian. If calling directly Jr should be passed as np.eye(3)
        Jl -- If specified it computes the left jacobian. If calling directly Jr should be passed as np.eye(3)

        Returns:
        The inverse of the current SO3 instance
        and the associated jacobian if specified
        """
        if not Jr is None:
            return SO3(self.arr.T), -self.Adj @ Jr
        elif not Jl is None:
            R_inv = SO3(self.arr.T)
            return R_inv, -R_inv.Adj @ Jl
        else:
            return SO3(self.arr.T)

    def transpose(self) -> 'SO3':
        """Returns the transpose of the current instance"""
        return SO3(self.arr.T)

    def rota(self, v: np.ndarray, Jr: np.ndarray = None, Jl: np.ndarray = None) -> np.ndarray:
        """
        Computes the rotated vector by doing R * v

        Args:
        v -- Numpy array containing the vector to be rotated

        Keyword Args:
        Jr -- If specified it computes the right jacobian. If calling directly Jr should be passed as np.eye(3)
        Jl -- If specified it computes the left jacobian. If calling directly Jr should be passed as np.eye(3)

        Returns:
        The rotated vector as a numpy array
        """
        assert v.size == 3
        vp = self.R @ v
        if not Jr is None:
            J = -self.R @ skew(v)
            return vp, J @ Jr
        elif not Jl is None:
            J = -skew(vp)
            return vp, J @ Jl
        else:
            return vp

    def rotp(self, v: np.ndarray, Jr: np.ndarray = None, Jl: np.ndarray = None) -> np.ndarray:
        """
        Computes the rotated vector by doing R^-1 * v

        Args:
        v -- Numpy array containing the vector to be rotated

        Keyword Args:
        Jr -- If specified it computes the right jacobian. If calling directly Jr should be passed as np.eye(3)
        Jl -- If specified it computes the left jacobian. If calling directly Jr should be passed as np.eye(3)

        Returns:
        The rotated vector as a numpy array
        """
        assert v.size == 3
        if not Jr is None:
            R_inv, J = self.inv(Jr=Jr)
            vp, J = R_inv.rota(v, Jr=J)
            return vp, J
        elif not Jl is None:
            R_inv, J = self.inv(Jl=Jl)
            vp, J = R_inv.rota(v, Jl=J)
            return vp, J
        else:
            return self.inv().rota(v)

    # Assumes jacobian is with respect to v
    def boxplusr(self, v: np.ndarray, Jr: np.ndarray = None,
                 Jl: np.ndarray = None) -> 'SO3':
        """
        Computes self * Exp(v)

        Args:
        v -- The perturbation in the tangent space

        Keyword Args:
        Jr -- If specified it computes the right jacobian. If calling directly Jr should be passed as np.eye(3)
        Jl -- If specified it computes the left jacobian. If calling directly Jr should be passed as np.eye(3)

        Returns:
        An instance of SO3
        """
        assert(v.size == 3)
        if not Jr is None:
            R, J = SO3.Exp(v, Jr=Jr)
            res, J = self.compose(R, Jr2=J)
            return res, J
        elif not Jl is None:
            R, J = SO3.Exp(v, Jl=Jl)
            res, J = self.compose(R, Jl2=J)
            return res, J
        else:
            return self * SO3.Exp(v)

    def boxminusr(self, R2: 'SO3', Jr1: np.ndarray = None,
                  Jl1: np.ndarray = None, Jr2: np.ndarray = None,
                  Jl2: np.ndarray = None) -> np.ndarray:
        """
        Returns Log(R2.inv() * self)

        Args:
        R2 -- An instance of SO2

        Keyword Args:
        Jr1-- If specified it computes the right jacobian of self. If calling directly Jr should be passed as np.eye(3)
        Jl1 -- If specified it computes the left jacobian of self. If calling directly Jr should be passed as np.eye(3)
        Jr2 -- If specified it computes the right jacobian of R2. If calling directly Jr should be passed as np.eye(3)
        Jl2 -- If specified it computes the left jacobian of R2. If calling directly Jr should be passed as np.eye(3)

        Returns:
        A numpy array representing the difference in the tangent space.
        """
        assert isinstance(R2, SO3)
        if Jr1 is not None:
            dR, J = R2.inv().compose(self, Jr2=Jr1)
            return SO3.Log(dR, Jr=J)
        elif Jl1 is not None:
            dR, J = R2.inv().compose(self, Jl2=Jl1)
            return SO3.Log(dR, Jl=J)
        elif Jr2 is not None:
            R2_inv, J = R2.inv(Jr=Jr2)
            dR, J = R2_inv.compose(self, Jr=J)
            return SO3.Log(dR, Jr=J)
        elif Jl2 is not None:
            R2_inv, J = R2.inv(Jl=Jl2)
            dR, J = R2_inv.compose(self, Jl=J)
            return SO3.Log(dR, Jl=J)
        else:
            return SO3.Log(R2.inv() * self)

    def boxplusl(self, v: np.ndarray, Jr: np.ndarray = None,
                 Jl: np.ndarray = None) -> 'SO3':
        """
        Returns Exp(v) * self

        Args:
        v -- Numpy array for the perturbation vector in the tangent space

        Keyword Args:
        Jr -- If specified it computes the right jacobian. If calling directly Jr should be passed as np.eye(3)
        Jl -- If specified it computes the left jacobian. If calling directly Jr should be passed as np.eye(3)

        Returns:
        An instance of SO3
        """
        assert(v.size == 3)
        if Jr is not None:
            R, J = SO3.Exp(v, Jr=Jr)
            return R.compose(self, Jr=J)
        elif Jl is not None:
            R, J = SO3.Exp(v, Jl=Jl)
            return R.compose(self, Jl=J)
        else:
            return SO3.Exp(v) * self

    def boxminusl(self, R2: 'SO3', Jr1: np.ndarray = None,
                  Jl1: np.ndarray = None, Jr2: np.ndarray = None,
                  Jl2: np.ndarray = None):
        """
        Returns Log(R2.inv() * self)

        Args:
        R2 -- An instance of SO2

        Keyword Args:
        Jr1-- If specified it computes the right jacobian of self. If calling directly Jr should be passed as np.eye(3)
        Jl1 -- If specified it computes the left jacobian of self. If calling directly Jr should be passed as np.eye(3)
        Jr2 -- If specified it computes the right jacobian of R2. If calling directly Jr should be passed as np.eye(3)
        Jl2 -- If specified it computes the left jacobian of R2. If calling directly Jr should be passed as np.eye(3)

        Returns:
        A numpy array representing the difference in the tangent space.
        """

        assert isinstance(R2, SO3)
        if Jr1 is not None:
            diff, J = self.compose(R2.inv(), Jr=Jr1)
            return SO3.Log(diff, Jr=J)
        elif Jl1 is not None:
            diff, J = self.compose(R2.inv(), Jl=Jl1)
            return SO3.Log(diff, Jl=J)
        elif Jr2 is not None:
            R_inv, J = R2.inv(Jr=Jr2)
            diff, J = self.compose(R_inv, Jr2=J)
            return SO3.Log(diff, Jr=J)
        elif Jl2 is not None:
            R_inv, J = R2.inv(Jl=Jl2)
            diff, J = self.compose(R_inv, Jl2=J)
            return SO3.Log(diff, Jl=J)
        else:
            return SO3.Log(self * R2.inv())

    def normalize(self) -> None:
        """Normalizes the rotation matrix"""
        x = self.R[:,0]
        x = x / np.linalg.norm(x)
        y = np.cross(self.R[:,2], x)
        y = y / np.linalg.norm(y)
        z = np.cross(x, y)

        self.arr = np.array([[*x], [*y], [*z]]).T

    def det(self) -> float:
        """Computes the determinant"""
        return np.linalg.det(self.R)

    def compose(self, R: 'SO3', Jr: np.ndarray = None,
                Jl: np.ndarray = None, Jr2: np.ndarray = None,
                Jl2: np.ndarray = None):
        """
        Alternate way to * operator to compose two matrices. This allows for calculating the jacobians

        Args:
        R -- An instance of SO3

        Keyword Args:
        Jr1-- If specified it computes the right jacobian of self. If calling directly Jr should be passed as np.eye(3)
        Jl1 -- If specified it computes the left jacobian of self. If calling directly Jr should be passed as np.eye(3)
        Jr2 -- If specified it computes the right jacobian of R2. If calling directly Jr should be passed as np.eye(3)
        Jl2 -- If specified it computes the left jacobian of R2. If calling directly Jr should be passed as np.eye(3)

        Returns:
        An instance of SO3
        """
        res = self * R
        if not Jr is None:
            J = R.inv().Adj
            return res, J @ Jr
        elif not Jl is None:
            return res, np.eye(3) @ Jl
        elif not Jr2 is None:
            return res, np.eye(3) @ Jr2
        elif not Jl2 is None:
            return res, self.Adj @ Jl2
        else:
            return res

    @property
    def R(self) -> np.ndarray:
        """Returns the underlying array"""
        return self.arr

    @property
    def euler(self) -> np.ndarray:
        """Returns the RPY angles for the rotation matrix"""
        if 1 - np.abs(self.R[2,0]) > 1e-8:
            phi = np.arctan2(self.R[2,1], self.R[2,2])
            theta = np.arcsin(-self.R[2,0])
            psi = np.arctan2(self.R[1,0], self.R[0,0])
        else:
            phi = 0
            if np.sign(self.R[2,0]) > 0:
                theta = np.pi/2
                psi = -np.arctan2(-self.R[1,2], self.R[1,1])
            else:
                theta = -np.pi/2
                psi = np.arctan2(-self.R[1,2], self.R[1,1])
        return np.array([phi, theta, psi])

    @classmethod
    def fromRPY(cls, angles: np.ndarray) -> 'SO3':
        """
        Creates an SO3 instance from roll, pitch, yaw angles

        Args:
        angles -- A numpy array with [roll, pitch, yaw]

        Returns:
        An instance of SO3
        """
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
    def fromAxisAngle(cls, w: np.ndarray) -> 'SO3':
        """
        Creates an instance of SO3 from an Axis Angle Formulation

        Args:
        w -- A vector with norm of theta in the direction of the vector of rotation

        Returns:
        An instance of SO3
        """
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
    def fromQuaternion(cls, q: np.ndarray) -> 'SO3':
        """
        Creates an instance of SO3 from a quaternion

        Args:
        q -- A numpy array representing a quaternion in [qw, qx, qy, qz]

        Returns:
        An instance of SO3
        """
        qw = q[0]
        qv = q[1:]
        qv_skew = np.array([[0, -qv[2], qv[1]], [qv[2], 0, -qv[0]], [-qv[1], qv[0], 0]])

        R = (2 * qw**2 - 1) * np.eye(3) + 2 * qw * qv_skew + 2 * np.outer(qv, qv)
        return cls(R)

    @classmethod
    def random(cls) -> 'SO3':
        """
        Returns a randomly generated rotation matrix
        """
        x = np.random.uniform(0, 1, size=3)
        psi = 2 * np.pi * x[0]
        R = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        v = np.array([np.cos(2 * np.pi * x[1]) * np.sqrt(x[2]),
                     np.sin(2 * np.pi * x[1]) * np.sqrt(x[2]),
                     np.sqrt(1 - x[2])])
        H = np.eye(3) - 2 * np.outer(v, v)
        return cls(-H @ R)

    @staticmethod
    def Identity() -> 'SO3':
        """Returns the identity matrix"""
        return SO3(np.eye(3))

    @staticmethod
    def log(R: 'SO3', Jr: np.ndarray = None, Jl: np.ndarray = None) -> np.ndarray: #This function isn't entirely stable but tests pass
        """
        Performs the logarithmic map to convert SO3 elements to its tangent space

        Args:
        R -- An instance of SO3

        Keyword Args:
        Jr-- If specified it computes the right jacobian of self. If calling directly Jr should be passed as np.eye(3)
        Jl -- If specified it computes the left jacobian of self. If calling directly Jr should be passed as np.eye(3)

        Returns:
        The representation in the tangent space as a skew symmetric matrix
        """
        assert isinstance(R, SO3)

        theta = np.arccos((np.trace(R.arr) - 1)/2.0)
        if np.abs(theta) < 1e-8: # Do taylor series expansion
            temp = 1/2.0 * (1 + theta**2 / 6.0 + 7 * theta**4 / 360)
            logR = temp * (R - R.transpose())
        elif np.abs(np.abs(theta) - np.pi) < 1e-3:
            temp = - np.pi/(theta - np.pi) - 1 - np.pi/6 * (theta - np.pi) - (theta - np.pi)**2/6 - 7*np.pi/360 * (theta - np.pi)**3 - 7/360.0 * (theta - np.pi)**4
            logR = temp/2.0 * (R - R.transpose())
        else:
            logR = theta / (2.0 * np.sin(theta)) * (R - R.transpose())

        if not Jr is None: # TODO: Add Taylor series expansion?
            thetax = skew(SO3.vee(logR))
            J = np.eye(3) + 0.5 * thetax + (1/theta**2 - (1 + np.cos(theta))/(2 * theta * np.sin(theta))) * (thetax @ thetax)
            return logR, J @ Jr
        elif not Jl is None:
            thetax = skew(SO3.vee(logR))
            J = np.eye(3) - 0.5 * thetax + (1/theta**2 - (1 + np.cos(theta))/(2 * theta * np.sin(theta))) * (thetax @ thetax)
            return logR, J @ Jl
        else:
            return logR

    @classmethod # Make this method the above
    def Log(cls, R, Jr=None, Jl=None): #easy call to go straight to a vector
        if not Jr is None:
            logR, J = cls.log(R, Jr=Jr)
            return cls.vee(logR), J
        elif not Jl is None:
            logR, J = cls.log(R, Jl=Jl)
            return cls.vee(logR), J
        else:
            logR = cls.log(R)
            return cls.vee(logR)

    @classmethod
    def exp(cls, logR: np.ndarray, Jr: np.ndarray = None,
            Jl: np.ndarray = None) -> 'SO3':
        """
        Computes the exponential map to convert elements of the tangent space to the manifold

        Args:
        logR -- A skew symmetric matrix of the tangent space vector

        Keyword Args:
        Jr-- If specified it computes the right jacobian of self. If calling directly Jr should be passed as np.eye(3)
        Jl -- If specified it computes the left jacobian of self. If calling directly Jr should be passed as np.eye(3)

        Returns:
        An instance of SO3
        """
        assert logR.shape == (3,3)

        w = cls.vee(logR)
        theta = np.sqrt(w @ w)
        if np.abs(theta) > 1e-8:
            R = np.eye(3) + np.sin(theta)/theta * logR + (1 - np.cos(theta))/ (theta**2) * (logR @ logR)
        else: # Do taylor series expansion for small thetas
            R = np.eye(3)

        if not Jr is None: # Possibly add taylor series logic
            wx = skew(w)
            a = (1 - np.cos(theta)) / theta**2
            b = (theta - np.sin(theta)) / theta**3
            J = np.eye(3) - a * wx + b * (wx @ wx)
            return cls(R), J @ Jr
        elif not Jl is None:
            wx = skew(w)
            a = (1 - np.cos(theta)) / theta**2
            b = (theta - np.sin(theta)) / theta**3
            J = np.eye(3) + a * wx + b * (wx @ wx)
            return cls(R), J @ Jl
        else:
            return cls(R)

    @classmethod # Do this in the above method
    def Exp(cls, w, Jr=None, Jl=None):
        logR = cls.hat(w)
        if not Jr is None:
            R, J = cls.exp(logR, Jr=Jr)
            return R, J
        elif not Jl is None:
            R, J = cls.exp(logR, Jl=Jl)
            return R, J
        else:
            return cls.exp(logR)

    @staticmethod
    def vee(logR: np.ndarray) -> np.ndarray:
        """
        Convert a skew symmetric matrix to a vector

        Args:
        logR -- A skew symmetric matrix

        Returns:
        The vector used to make logR
        """
        assert logR.shape == (3,3)
        omega = np.array([logR[2,1], logR[0,2], logR[1,0]])
        return omega

    @staticmethod
    def hat(omega: np.ndarray) -> np.ndarray:
        """
        Creates a skew symmetric matrix from a 3 array

        Args:
        omega -- A numpy array of size 3

        Returns:
        A 3x3 skew symmetric matrix
        """
        assert omega.size == 3
        return (G @ omega).squeeze()

    @property
    def Adj(self) -> np.ndarray:
        """Computes the Adjoint for the group instance"""
        return self.arr
