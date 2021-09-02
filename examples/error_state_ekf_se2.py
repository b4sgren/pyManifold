import numpy as np
import sys
sys.path.append("..")
from se2 import SE2
from se2_classes import Robot, SE2_EKF
import matplotlib.pyplot as plt

if __name__=="__main__":
  t = 0.0
  tf = 60.0
  dt = 0.1

  true_robot = Robot()
  est_robot = Robot()

  state_hist = []
  est_hist = []
  state_hist.append([true_robot.state.x, true_robot.state.y,\
                    true_robot.state.theta])
  est_hist.append([est_robot.state.x, est_robot.state.y,\
                    est_robot.state.theta])

  t_hist = [t]

  while t < tf:
    u, u_hat = true_robot.getInputs(t)
    true_robot.propogateDynamics(u, dt)
    est_robot.propogateDynamics(u_hat, dt)

    t += dt
    state_hist.append([true_robot.state.x, true_robot.state.y, \
                      true_robot.state.theta])
    est_hist.append([est_robot.state.x, est_robot.state.y, \
                      est_robot.state.theta])

    t_hist.append(t)

  state_hist = np.array([state_hist]).T
  est_hist = np.array([est_hist]).T
  fig, ax = plt.subplots(3,1)
  ax[0].plot(t_hist, state_hist[0])
  ax[0].plot(t_hist, est_hist[0])
  ax[1].plot(t_hist, state_hist[1])
  ax[1].plot(t_hist, est_hist[1])
  ax[2].plot(t_hist, state_hist[2])
  ax[2].plot(t_hist, est_hist[2])

  plt.figure(2)
  plt.plot(state_hist[0], state_hist[1])
  plt.plot(est_hist[0], est_hist[1])

  plt.show()
