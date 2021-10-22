import numpy as np
import sys
sys.path.append("..")
from quaternion import Quaternion as Quat
import matplotlib.pyplot as plt
from trajectory import QuadPrams, Trajectory

class Quadrotor:
    def __init__(self, quad_params, traj):
        self.mass = quad_params.mass
        self.J = quad_params.J

        p,v, _, R_i_from_b, _ = traj.calcStep(0)
        # Quadrotor state
        # self.position = np.array([3.0, 0, -5.0])
        # self.velocity = np.array([0, .62831853, 0])
        # self.q_i_from_b = Quat.Identity()
        self.position = p
        self.velocity = v
        self.q_i_from_b = Quat.fromRotationMatrix(R_i_from_b.R)

        # Uncertainty
        self.P_ = np.zeros((9,9))

        self.g = 9.81

    def propogateDynamics(self, ab, wb, dt):
        xdot = self.velocity
        vdot = np.cross(v, wb) + ab

        # Propagate state
        self.position += xdot * dt
        self.velocity += vdot * dt
        self.q_i_from_b = self.q_i_from_b.boxplusr(wb*dt)

        # Propagate Uncertainty

class EKF:
    def __init__(self):
        self.R_accel = np.diag([1e-3, 1e-3, 1e-3])
        self.R_gyro = np.diag([1e-3, 1e-3, 1e-3])

if __name__=="__main__":
    t0 = 0.0
    tf = 60.0
    dt = 0.01 # IMU update rate of 100Hz

    params = QuadPrams(1.0)
    traj = Trajectory(params, True)
    quad = Quadrotor(params, traj)
    truth_quad = Quadrotor(params, traj)
    dr_quad = Quadrotor(params, traj)
    ekf = EKF()

    t_hist = np.arange(t0, tf, dt)

    x_hist, v_hist, euler_hist = [], [], []
    truth_x_hist, truth_v_hist, truth_euler_hist = [], [], []
    dr_x_hist, dr_v_hist, dr_euler_hist = [], [], []
    for t in t_hist:
        pos, v, ab, R_i_from_b, wb = traj.calcStep(t)
        eta_a = np.random.multivariate_normal(np.zeros(3), ekf.R_accel)
        eta_g = np.random.multivariate_normal(np.zeros(3), ekf.R_gyro)
        truth_quad.propogateDynamics(ab, wb, dt)
        quad.propogateDynamics(ab+eta_a, wb+eta_g, dt)
        dr_quad.propogateDynamics(ab+eta_a, wb+eta_g, dt)

        x_hist.append(quad.position.copy())
        v_hist.append(quad.velocity.copy())
        euler_hist.append(quad.q_i_from_b.euler)
        truth_x_hist.append(truth_quad.position.copy())
        truth_v_hist.append(truth_quad.velocity.copy())
        truth_euler_hist.append(truth_quad.q_i_from_b.euler)
        dr_x_hist.append(dr_quad.position.copy())
        dr_v_hist.append(dr_quad.velocity.copy())
        dr_euler_hist.append(dr_quad.q_i_from_b.euler)



    x_hist = np.array(x_hist).T
    v_hist = np.array(v_hist).T
    euler_hist = np.array(euler_hist).T
    truth_x_hist = np.array(truth_x_hist).T
    truth_v_hist = np.array(truth_v_hist).T
    truth_euler_hist = np.array(truth_euler_hist).T
    dr_x_hist = np.array(dr_x_hist).T
    dr_v_hist = np.array(dr_v_hist).T
    dr_euler_hist = np.array(dr_euler_hist).T


    fig1, ax1 = plt.subplots(nrows=3, ncols=1)
    ax1[0].plot(t_hist, truth_x_hist[0], label='Truth')
    ax1[1].plot(t_hist, truth_x_hist[1])
    ax1[2].plot(t_hist, truth_x_hist[2])
    ax1[0].plot(t_hist, x_hist[0], label='Est')
    ax1[1].plot(t_hist, x_hist[1])
    ax1[2].plot(t_hist, x_hist[2])
    ax1[0].plot(t_hist, dr_x_hist[0], label='DR')
    ax1[1].plot(t_hist, dr_x_hist[1])
    ax1[2].plot(t_hist, dr_x_hist[2])
    ax1[0].set_title("Positions vs Time")
    ax1[0].legend()

    fig2, ax2 = plt.subplots(nrows=3, ncols=1)
    ax2[0].plot(t_hist, truth_v_hist[0], label='Truth')
    ax2[1].plot(t_hist, truth_v_hist[1])
    ax2[2].plot(t_hist, truth_v_hist[2])
    ax2[0].plot(t_hist, v_hist[0], label='Est')
    ax2[1].plot(t_hist, v_hist[1])
    ax2[2].plot(t_hist, v_hist[2])
    ax2[0].plot(t_hist, dr_v_hist[0], label='DR')
    ax2[1].plot(t_hist, dr_v_hist[1])
    ax2[2].plot(t_hist, dr_v_hist[2])
    ax2[0].set_title("Velocity vs Time")
    ax2[0].legend()

    fig3, ax3 = plt.subplots(nrows=3, ncols=1)
    ax3[0].plot(t_hist, truth_euler_hist[0], label='Truth')
    ax3[1].plot(t_hist, truth_euler_hist[1])
    ax3[2].plot(t_hist, truth_euler_hist[2])
    ax3[0].plot(t_hist, euler_hist[0], label='Est')
    ax3[1].plot(t_hist, euler_hist[1])
    ax3[2].plot(t_hist, euler_hist[2])
    ax3[0].plot(t_hist, dr_euler_hist[0], label='DR')
    ax3[1].plot(t_hist, dr_euler_hist[1])
    ax3[2].plot(t_hist, dr_euler_hist[2])
    ax3[0].set_title("Euler Angles vs Time")
    ax3[0].legend()

    fig4, ax4 = plt.subplots(nrows=1, ncols=1)
    ax4.plot(truth_x_hist[0], truth_x_hist[1], label='Truth')
    ax4.plot(x_hist[0], x_hist[1], label='Est')
    ax4.plot(dr_x_hist[0], dr_x_hist[1], label='DR')
    ax4.legend()

    plt.show()
