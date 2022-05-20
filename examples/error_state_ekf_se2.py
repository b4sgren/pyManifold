import numpy as np
import sys
sys.path.append("..")
from pyManifold.se2 import SE2
from se2_classes import Robot, SE2_EKF
import matplotlib.pyplot as plt

if __name__=="__main__":
  t = 0.0
  tf = 60.0
  dt = 0.1
  lms = [np.array([0, -5.0]), np.array([7.0, 0]), np.array([-5.0, -10.0])]

  true_robot = Robot()
  est_robot = Robot()
  dr_robot = Robot()
  ekf = SE2_EKF()

  state_hist = []
  est_hist = []
  dr_hist = []
  state_hist.append([true_robot.state.x, true_robot.state.y,\
                    true_robot.state.theta])
  est_hist.append([est_robot.state.x, est_robot.state.y,\
                    est_robot.state.theta])
  dr_hist.append([dr_robot.state.x, dr_robot.state.y,\
                    dr_robot.state.theta])


  t_hist = [t]

  while t < tf:
    u, u_hat = true_robot.getInputs(t)
    true_robot.propogateDynamics(u, dt)
    est_robot.propogateDynamics(u_hat, dt)
    dr_robot.propogateDynamics(u_hat, dt)
    ekf.propogateDynamics(est_robot, u_hat, dt)

    z = true_robot.measurements(lms)
    est_robot = ekf.measurementUpdate(est_robot, z, lms)

    t += dt
    state_hist.append([true_robot.state.x, true_robot.state.y, \
                      true_robot.state.theta])
    est_hist.append([est_robot.state.x, est_robot.state.y, \
                      est_robot.state.theta])
    dr_hist.append([dr_robot.state.x, dr_robot.state.y, \
                      dr_robot.state.theta])


    t_hist.append(t)

  state_hist = np.array([state_hist]).T
  est_hist = np.array([est_hist]).T
  dr_hist = np.array([dr_hist]).T
  fig, ax = plt.subplots(3,1)
  ax[0].plot(t_hist, state_hist[0])
  ax[0].plot(t_hist, est_hist[0])
  ax[0].plot(t_hist, dr_hist[0])
  ax[1].plot(t_hist, state_hist[1])
  ax[1].plot(t_hist, est_hist[1])
  ax[1].plot(t_hist, dr_hist[1])
  ax[2].plot(t_hist, state_hist[2])
  ax[2].plot(t_hist, est_hist[2])
  ax[2].plot(t_hist, dr_hist[2])

  plt.figure(2)
  plt.plot(state_hist[0], state_hist[1])
  plt.plot(est_hist[0], est_hist[1])
  plt.plot(dr_hist[0], dr_hist[1])

  plt.show()
