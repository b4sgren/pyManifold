import numpy as np
import sys
sys.path.append("..")
from se2 import SE2

class Robot:
  def __init__(self):
    self._state = SE2.Identity()

  @property
  def state(self):
    return SE2.Log(self._state)

  def getInputs(self, t):
    v = 1 #+ .5 * np.cos(2*np.pi*0.2*t)
    w = -0.2 #+ 2*np.cos(2*np.pi*0.6*t)
    return np.array([v,w])

  def propogateDynamics(self, u, dt):
    v,w = u[0], u[1]
    theta = SE2.Log(self._state)[2]

    dx = v * np.cos(theta) * dt
    dy = v * np.sin(theta) * dt
    dphi = w * dt
    t = self._state.inv().R @ np.array([dx, dy])
    dstate = SE2.fromAngleAndt(dphi, t)
    self._state = self._state * dstate

class SE2_EKF:
  def __init__(self, lms):
    self._cov = np.zeros((3,3))
    self.lms = lms

  def update(self, robot, u, z):
    pass

  def propogateDynamics(self, u):
    pass # Not sure I'll need this.

  def measurementUpdate(self, z):
    pass
