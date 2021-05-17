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

  state_hist = [true_robot.state]
  t_hist = [t]

  while t < tf:
    u = true_robot.getInputs(t)
    true_robot.propogateDynamics(u, dt)

    t += dt
    state_hist.append(true_robot.state)
    t_hist.append(t)

  state_hist = np.array([state_hist]).T
  fig, ax = plt.subplots(3,1)
  ax[0].plot(t_hist, state_hist[0])
  ax[1].plot(t_hist, state_hist[1])
  ax[2].plot(t_hist, state_hist[2])

  plt.figure(2)
  plt.plot(state_hist[0], state_hist[1])

  plt.show()
