import numpy as np
from typing import Tuple, Iterable

G = np.array([[0, -1], [1, 0]])

class SO2:
    def __init__(self, R: np.ndarray) -> None:
        """
        Constructor
        Args:
        R -- 2x2 rotation matrix
        """
        assert R.shape == (2,2)
        self.arr = R

    def __mul__(self, R2:'SO2') -> 'SO2':
        """
        Multiplication Operator

        Args:
        R2 -- SO2 object

        Returns:
        Instance of SO2
        """
        assert isinstance(R2, SO2)
        return SO2(self.arr @ R2.arr)

    def __str__(self):
        return str(self.R)

    def __repr__(self):
        return str(self.R)

    def inv(self, Jr: int = None, Jl: int = None) -> 'SO2':
        """
        Take the inverse of the SO2 element

        Keyword Args:
        Jr -- Indicates if the right jacobian is to be computed. Pass 1 if calling directly
        Jl -- Indicates if the left jacobian is to be computed. Pass 1 if calling directly

        Returns:
        Instance of SO2
        If Jr or Jl is specified it will also return the associated jacobian
        """
        if Jr:
            J = -1
            return SO2(self.arr.T), J * Jr
        elif Jl:
            J = -1
            return SO2(self.arr.T), J * Jl
        else:
            return SO2(self.arr.T)

    def rotate(self, v: np.ndarray, Jr: int = None, Jl: int = None) -> np.ndarray:
        """
        Rotates an element using R

        Args:
        v -- The vector to be rotated

        Keyword Args:
        Jr -- Indicates if the right jacobian is to be computed. Pass 1 if calling directly
        Jl -- Indicates if the left jacobian is to be computed. Pass 1 if calling directly

        Returns:
        Rotated vector as numpy array
        If Jr or Jl is specified it will return the associated jacobian
        """
        assert v.size == 2
        if Jr:
            J = self.R @ G @ v
            return self.R @ v, J * Jr
        elif Jl:
            J = G @ self.R @ v
            return self.R @ v, J * Jl
        else:
            return self.R @ v

    def inv_rotate(self, v, Jr=None, Jl=None) -> np.ndarray:
        """
        Rotates an element using R.inv()

        Args:
        v -- The vector to be rotated

        Keyword Args:
        Jr -- Indicates if the right jacobian is to be computed. Pass 1 if calling directly
        Jl -- Indicates if the left jacobian is to be computed. Pass 1 if calling directly

        Returns:
        Rotated vector as numpy array
        If Jr or Jl is specified it will return the associated jacobian
        """
        assert v.size == 2
        if Jr:
            R_inv, J = self.inv(Jr=Jr)
            vp, J = R_inv.rotate(v, Jr=J)
            return vp, J
        elif Jl:
            R_inv, J = self.inv(Jl=Jl)
            vp, J = R_inv.rotate(v, Jl=J)
            return vp, J
        else:
            return self.inv().rotate(v)

    # For the boxplus I'm not sure I need the jacobian wrt the first element ever. I can add it later if I need to
    def boxplusr(self, w:float, Jr: int = None, Jl: int = None) -> 'SO2':
        """
        Peforms a generalized plus operation: R * Exp(w)

        Args:
        w -- A scalar in the tangent space of SO2

        Keyword Args:
        Jr -- Indicates if the right jacobian is to be computed. Pass 1 if calling directly
        Jl -- Indicates if the left jacobian is to be computed. Pass 1 if calling directly

        Returns:
        Instance of SO2
        If Jr or Jl is specified it will return the associated Jacobian
        """
        if Jr:
            R2, J = SO2.Exp(w, Jr=Jr)
            return self.compose(R2, Jr2=J)
        elif Jl:
            R2, J = SO2.Exp(w, Jl=Jl)
            return self.compose(R2, Jl2=J)
        else:
            return self * SO2.Exp(w)

    def boxminusr(self, R2: 'SO2', Jr1: int = None, Jl1: int = None,
                  Jr2: int = None, Jl2: int = None) -> float:
        """
        Peforms a generalized minus operation: Log(R2.inv() * self)

        Args:
        R2 -- An SO2 instance

        Keyword Args:
        Jr1 -- Indicates if the right jacobian is to be computed wrt self. Pass 1 if calling directly
        Jl1 -- Indicates if the left jacobian is to be computed wrt self. Pass 1 if calling directly
        Jr2 -- Indicates if the right jacobian is to be computed wrt R2. Pass 1 if calling directly
        Jl2 -- Indicates if the left jacobian is to be computed wrt R2. Pass 1 if calling directly

        Returns:
        The angle representing the difference between self and R2
        If a Jacobian is specified it will also return the corresponding jacobian
        """
        assert isinstance(R2, SO2)
        if Jr1 is not None:
            dR, J = R2.inv().compose(self, Jr2=Jr1)
            return SO2.Log(dR, Jr=J)
        elif Jl1 is not None:
            dR, J = R2.inv().compose(self, Jl2=Jl1)
            return SO2.Log(dR, Jl=J)
        elif Jr2 is not None:
            R2_inv, J = R2.inv(Jr=Jr2)
            dR, J = R2_inv.compose(self, Jr=J)
            return SO2.Log(dR, Jr=J)
        elif Jl2 is not None:
            R2_inv, J = R2.inv(Jl=Jl2)
            dR, J = R2.inv().compose(self, Jl=J)
            return SO2.Log(dR, Jl=J)
        else:
            return SO2.Log(R2.inv() * self)

    def boxplusl(self, w: float, Jr: int = None, Jl: int = None) -> 'SO2':
        """
        Peforms a generalized plus operation: Exp(w) * R

        Args:
        w -- A scalar in the tangent space of SO2

        Keyword Args:
        Jr -- Indicates if the right jacobian is to be computed. Pass 1 if calling directly
        Jl -- Indicates if the left jacobian is to be computed. Pass 1 if calling directly

        Returns:
        An instance of SO2
        If Jr or Jl is specified it will also return the associated jacobian
        """
        if Jr is not None:
            R, J = SO2.Exp(w, Jr=Jr)
            return R.compose(self, Jr=J)
        elif Jl is not None:
            R, J = SO2.Exp(w, Jl=Jl)
            return R.compose(self, Jl=J)
        else:
            return SO2.Exp(w) * self

    def boxminusl(self, R: 'SO2', Jr1: int = None, Jl1: int = None,
                  Jr2: int = None, Jl2: int = None) -> float:
        """
        Peforms a generalized minus operation: Log(self * R2.inv())

        Args:
        R2 -- An SO2 instance

        Keyword Args:
        Jr1 -- Indicates if the right jacobian is to be computed wrt self. Pass 1 if calling directly
        Jl1 -- Indicates if the left jacobian is to be computed wrt self. Pass 1 if calling directly
        Jr2 -- Indicates if the right jacobian is to be computed wrt R2. Pass 1 if calling directly
        Jl2 -- Indicates if the left jacobian is to be computed wrt R2. Pass 1 if calling directly

        Returns:
        A float representing the difference between self and R
        If any of the Jacobians are specified it will return the corresponding jacobian as well
        """
        assert isinstance(R, SO2)
        if Jr1 is not None:
            temp, J = self.compose(R.inv(), Jr=Jr1)
            return SO2.Log(temp, Jr=J)
        elif Jl1 is not None:
            temp, J = self.compose(R.inv(), Jl=Jl1)
            return SO2.Log(temp, Jl=J)
        elif Jr2 is not None:
            R_inv, J = R.inv(Jr=Jr2)
            temp, J = self.compose(R_inv, Jr2=J)
            return SO2.Log(temp, Jr=J)
        elif Jl2 is not None:
            R_inv, J = R.inv(Jl=Jl2)
            temp, J = self.compose(R_inv, Jl2=J)
            return SO2.Log(temp, Jl=J)
        else:
            return SO2.Log(self * R.inv())

    def compose(self, R: 'SO2', Jr: int = None, Jl: int = None, Jr2: int = None, Jl2: int = None) -> 'SO2':
        """
        Alternate call to compose two rotation matrices. This call allows for jacobian calculation

        Args:
        R -- An SO2 instance

        Keyword Args:
        Jr1 -- Indicates if the right jacobian is to be computed wrt self. Pass 1 if calling directly
        Jl1 -- Indicates if the left jacobian is to be computed wrt self. Pass 1 if calling directly
        Jr2 -- Indicates if the right jacobian is to be computed wrt R. Pass 1 if calling directly
        Jl2 -- Indicates if the left jacobian is to be computed wrt R. Pass 1 if calling directly

        Returns:
        An instance of SO2
        If any of the jacobians are specified it will also return the corresponding jacobian
        """
        res = self * R
        if Jr:
            return res, R.inv().Adj * Jr
        elif Jr2:
            return res, R.inv().Adj * Jr2
        elif Jl:
            return res, 1.0 * Jl
        elif Jl2:
            return res, 1.0 * Jl2
        else:
            return res

    @property
    def R(self) -> np.ndarray:
        """Returns the underlying np.array"""
        return self.arr

    @property
    def theta(self) -> float:
        """ Returns the angle representing the rotation matrix"""
        return SO2.Log(self)

    @classmethod
    def fromAngle(cls, theta: float) -> 'SO2':
        """
        Creates an SO2 instance from an angle

        Args:
        theta -- The rotation angle

        Returns:
        An instance of SO2
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        R = np.array([[ct, -st], [st, ct]])
        return cls(R)

    @classmethod
    def exp(cls, theta_x: np.ndarray, Jr: int = None, Jl: int = None) -> 'SO2':
        """
        Performs the Exponential Map to do R^1 -> SO2

        Args:
        theta_x -- Skew symetric matrix

        Keyword Args:
        Jr -- Indicates if the right jacobian is to be computed. Pass 1 if calling directly
        Jl -- Indicates if the left jacobian is to be computed. Pass 1 if calling directly

        Returns:
        An instance of SO2
        If Jr or Jl is specified it also will return the associated jacobian
        """
        assert theta_x.shape == (2,2)
        theta = theta_x[1,0]
        if Jr:
            return cls.fromAngle(theta), 1.0 * Jr
        if Jl:
            return cls.fromAngle(theta), 1.0 * Jl
        return cls.fromAngle(theta)

    @classmethod # Get rid of this method
    def Exp(cls, theta, Jr=None, Jl=None):
        logR = cls.hat(theta)
        if Jr:
            R, J = cls.exp(logR, Jr=Jr)
            return R, J
        elif Jl:
            R, J = cls.exp(logR, Jl=Jl)
            return R, J
        else:
            return cls.exp(logR)

    @staticmethod
    def Identity() -> 'SO2':
        """Returns SO2 element at identity"""
        return SO2(np.eye(2))

    @staticmethod
    def log(R: 'SO2', Jr: int = None, Jl: int = None) -> float:
        assert isinstance(R, SO2)
        """
        Performs the Logarithmic Map to do SO2 -> R^1

        Args:
        R -- Instance of SO2

        Keyword Args:
        Jr -- Indicates if the right jacobian is to be computed. Pass 1 if calling directly
        Jl -- Indicates if the left jacobian is to be computed. Pass 1 if calling directly

        Returns:
        A float in the tangent space of SO2
        If Jr or Jl is specified it will also return the associated jacobian
        """
        theta = np.arctan2(R.arr[1,0], R.arr[0,0])
        if Jr:
            return G * theta, 1.0 * Jr
        elif Jl:
            return G * theta, 1.0 * Jl
        else:
            return G * theta

    @classmethod # Get rid of this
    def Log(cls, R, Jr=None, Jl=None):
        if Jr:
            logR, J = cls.log(R, Jr)
            return logR, J * Jr
        elif Jl:
            logR, J = cls.log(R,Jl)
            return logR, J * Jl
        else:
            logR = cls.log(R)
            return cls.vee(logR)

    @staticmethod
    def vee(theta_x: np.ndarray) -> float:
        """
        vee operator to convert a 2x2 skew symmetric matrix to a scalar

        Args:
        theta_x -- A 2x2 skew symmetric matrix

        Returns:
        The float used to form the skew symmetric matrix
        """
        assert theta_x.shape == (2,2)
        return theta_x[1,0]

    @staticmethod
    def hat(theta: float) -> np.ndarray:
        """
        hat operator to convert a scalar to a skew symmetric matrix

        Args:
        theta -- Angle to put into a 2x2 skew symmetric matrix

        Return:
        A skew symmetric matrix from the angle
        """
        return theta * G

    @property
    def Adj(self) -> float:
        """Adjoint operator"""
        return 1.0

    @classmethod
    def random(cls) -> 'SO2':
        """
        Create a random instance of SO2
        """
        theta = np.random.uniform(-np.pi, np.pi)
        return cls.fromAngle(theta)

# Class needs to be tested still. Are these equations valid for non-SE3. Check paper
class UncertainSO2(SO2):
    def __init__(self, R: np.ndarray, cov: np.ndarray) -> None:
        super().__init__(R)
        self.cov_ = cov

    def compose(self, T_jk: 'UncertainSO2', cov_ij: float = 0.0) -> 'UncertainSO2':
        T_ik = self * T_jk
        cov_ik = self.cov_ + self.Adj*T_jk.cov_*self.Adj + 2*self.Adj*cov_ij

        return UncertainSO2(T_ik.R, cov_ik)

    def inv(self) -> 'UncertainSO2':
        Tinv = super().inv()
        cov = Tinv.Adj * self.cov_ * Tinv.Adj
        return UncertainSO2(Tinv.R, cov)

    def between(self, T_ik: 'UncertainSO2', cov_ij: float = 0.0) -> 'UncertainSO2':
        T_inv = self.inv()
        T_jk = T_inv * T_ik # can I do this ...
        Adj_ti = T_inv.Adj
        cov = T_inv.cov_ + Adj_ti*T_ik.cov_*Adj_ti - 2*Adj_ti*cov_ij*Adj_ti

        return UncertainSO2(T_jk.R, cov)
