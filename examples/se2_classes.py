import numpy as np
import sys

from numpy.random import multivariate_normal
sys.path.append("..")
from pyManifold.se2 import SE2

class Robot:
  def __init__(self):
    self._state = SE2.Identity()
    self._odom_cov = np.diag([.1, .1, 1e-2])
    self._R = np.diag([1e-3, 1e-3])

  @property
  def state(self):
    return self._state

  @state.setter
  def state(self, x):
    self._state = x

  @property
  def odom_cov(self):
    return self._odom_cov

  @property
  def meas_cov(self):
    return self._R

  def getInputs(self, t):
    v = 1 + .5 * np.cos(2*np.pi*0.2*t)
    w = -0.2 + 2*np.cos(2*np.pi*0.6*t)

    u = np.array([v, 0, w])
    u_hat = u + np.random.multivariate_normal(np.zeros(3), self._odom_cov)

    return u, u_hat

  def propogateDynamics(self, u, dt):
    self._state = self._state.boxplusr(u*dt)

  def measurements(self, lms):
    # Currently measing position for simplicity.
    # Move to range/bearing after it is working
    z = [self.state.inv_transform(lm) + np.random.multivariate_normal(np.zeros(2), self._R) for lm in lms]
    return z

class SE2_EKF:
  def __init__(self):
    self._cov = np.zeros((3,3))

  def propogateDynamics(self, robot, u, dt):
    U, G = SE2.Exp(u*dt, Jr=np.eye(3))
    _, F = robot.state.compose(U, Jr=np.eye(3))

    self._cov = F @ self._cov @ F.T + G @ robot.odom_cov @ G.T

  def measurementUpdate(self, robot, z, lms):
    for zi, lm in zip(z, lms):
      T_inv, J1 = robot.state.inv(Jr=np.eye(3))
      z_hat, J2 = T_inv.transform(lm, Jr=np.eye(3))
      # Compose Jacobians according to chain rule
      H = J2 @ J1

      # Calculate Innovation
      y = zi - z_hat
      S = H @ self._cov @ H.T + robot.meas_cov

      # Kalman Gain
      K = self._cov @ H.T @ np.linalg.inv(S)

      # State correction
      dx = K @ y
      robot.state = robot.state.boxplusr(dx)
      M = np.eye(3) - K @ H
      self._cov = M @ self._cov @ M.T + K @ robot.meas_cov @ K.T

      return robot
