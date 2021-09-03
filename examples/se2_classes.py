import numpy as np
import sys
sys.path.append("..")
from se2 import SE2

class Robot:
  def __init__(self):
    self._state = SE2.Identity()
    self._odom_cov = np.diag([.1, .1, 1e-2])

  @property
  def state(self):
    return self._state

  def getInputs(self, t):
    v = 1 + .5 * np.cos(2*np.pi*0.2*t)
    w = -0.2 + 2*np.cos(2*np.pi*0.6*t)

    u = np.array([v, 0, w])
    u_hat = u + np.random.multivariate_normal(np.zeros(3), self._odom_cov)

    return u, u_hat

  def propogateDynamics(self, u, dt):
    self._state = self._state.boxplusr(u*dt)

class SE2_EKF:
  def __init__(self, lms):
    self._cov = np.zeros((3,3))
    self.lms = lms

  def update(self, robot, u, z):
    pass

  def propogateDynamics(self, state, u, dt):
    v, w = u[0], u[1]

    t = np.array([v*dt, 0])
    dphi = w*dt
    dT = SE2.fromAngleAndt(dphi, t)

  def measurementUpdate(self, z):
    pass
