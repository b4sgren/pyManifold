import numpy as np
import sys
sys.path.append("..")
from quaternion import Quaternion as Quat
import matplotlib.pyplot as plt
from trajectory import QuadPrams, Trajectory

class Quadrotor:
    def __init__(self, quad_params):
        self.mass = quad_params.mass
        self.J = quad_params.J

        # Quadrotor state
        self.position = np.array([3.0, 0, -5.0])
        self.velocity = np.array([0, .62831853, 0])
        self.q_i_from_b = Quat.Identity()

        # Uncertainty
        self.P_ = np.zeros((9,9))

        self.g = 9.81

    def propogateDynamics(self, ab, wb, dt):
        e3 = np.array([0, 0, 1])
        # xdot = self.q_i_from_b.rota(self.velocity)
        # vdot = np.cross(v, wb) + ab[2] + self.q_i_from_b.rota(self.g*e3)

        xdot = self.velocity
        vdot = np.cross(v, wb) + ab

        self.position += xdot * dt
        self.velocity += vdot * dt
        self.q_i_from_b = self.q_i_from_b.boxplusr(wb*dt)

if __name__=="__main__":
    t0 = 0.0
    tf = 60.0
    dt = 0.01 # IMU update rate of 100Hz

    params = QuadPrams(1.0)
    traj = Trajectory(params, True)
    quad = Quadrotor(params)

    t_hist = np.arange(t0, tf, dt)

    x_hist, v_hist, euler_hist = [], [], []
    for t in t_hist:
        pos, v, ab, R_i_from_b, wb = traj.calcStep(t)
        quad.propogateDynamics(ab, wb, dt)

        x_hist.append(quad.position.copy())
        v_hist.append(quad.velocity.copy())
        euler_hist.append(quad.q_i_from_b.euler)

    x_hist = np.array(x_hist).T
    v_hist = np.array(v_hist).T
    euler_hist = np.array(euler_hist).T

    fig1, ax1 = plt.subplots(nrows=3, ncols=1)
    ax1[0].plot(t_hist, x_hist[0])
    ax1[1].plot(t_hist, x_hist[1])
    ax1[2].plot(t_hist, x_hist[2])
    ax1[0].set_title("Positions vs Time")

    fig2, ax2 = plt.subplots(nrows=3, ncols=1)
    ax2[0].plot(t_hist, v_hist[0])
    ax2[1].plot(t_hist, v_hist[1])
    ax2[2].plot(t_hist, v_hist[2])
    ax2[0].set_title("Velocity vs Time")

    fig3, ax3 = plt.subplots(nrows=3, ncols=1)
    ax3[0].plot(t_hist, euler_hist[0])
    ax3[1].plot(t_hist, euler_hist[1])
    ax3[2].plot(t_hist, euler_hist[2])
    ax3[0].set_title("Euler Angles vs Time")

    fig4, ax4 = plt.subplots(nrows=1, ncols=1)
    ax4.plot(x_hist[0], x_hist[1])

    plt.show()
