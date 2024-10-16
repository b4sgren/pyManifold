import numpy as np
import sys
sys.path.append("..")
from pyManifold.quaternion import Quaternion as Quat, skew
from pyManifold.so3 import SO3
import matplotlib.pyplot as plt
from trajectory import QuadPrams, Trajectory
import scipy.linalg as spl

np.set_printoptions(linewidth=200, precision=4)

R_accel = np.diag([1e-3, 1e-3, 1e-3])
R_gyro = np.diag([1e-5, 1e-5, 1e-5])
R_alt = 1e-2
R_gps = np.diag([.25, .25, 1.0])
Q = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4])

class Quadrotor:
    def __init__(self, quad_params, traj, t0=0.0):
        self.mass = quad_params.mass
        self.J = quad_params.J

        p,v, _, R_i_from_b, _ = traj.calcStep(t0)
        # Quadrotor state position, body frame vel, quat
        self.position = p
        self.velocity = v
        # Do .T on rotation matrix because the matrix that is passed in is R(q)
        # self.q_i_from_b = Quat.fromRotationMatrix(R_i_from_b.R.T)
        self.q_i_from_b = Quat.fromRotationMatrix(R_i_from_b.R)

        # Uncertainty
        # self.P_ = np.zeros((9,9))
        # self.P_ = np.diag([4, 4, .1, 1, 1, .1, .5, .5, .5])
        self.P_ = np.diag([.1, .1, .1, .1, .1, .1, .1, .1, .1])

        self.g = 9.81

    def propogateDynamics(self, ab, wb, dt):
        e3 = np.array([0, 0, 1])
        xdot = self.q_i_from_b.rotate(self.velocity)
        vdot = skew(self.velocity) @ wb - self.q_i_from_b.inv_rotate(self.g*e3) + ab[2]*e3

        F, G = self.getOdomJacobians(ab, wb, dt)

        # Propagate state
        dx = np.block([xdot, vdot, wb]) * dt
        self.boxplusr(dx)

        # Discretize matrices
        Fd = np.eye(dx.size) + F*dt + F@F*(dt**2)/2
        Gd = G*dt

        # Propagate Uncertainy
        R = spl.block_diag(R_accel, R_gyro)
        # self.P_ = F @ self.P_ @ F.T + Q + G @ R @ G.T
        # self.P_ = Fd @ self.P_ @ Fd.T + Q*dt**2 + Gd @ R @ Gd.T
        self.P_ = Fd @ self.P_ @ Fd.T + Q + Gd @ R @ Gd.T

    def getOdomJacobians(self, ab, wb, dt):
        e3 = np.array([0, 0, 1])
        F = np.zeros((9,9))
        F[:3, 3:6] = self.q_i_from_b.R.T
        F[:3, 6:] = self.q_i_from_b.R.T @ skew(self.velocity)
        F[3:6, 3:6] = -skew(wb)
        F[3:6, 6:] = skew(self.q_i_from_b.R @ (self.g*e3))
        F[6:, 6:] = -skew(wb)

        G = np.zeros((9,6))
        G[3:6, 2] = e3
        G[3:6, 3:] = -skew(self.velocity)
        G[6:, 3:] = -np.eye(3)

        return F, G

    def boxplusr(self, dx):
        self.position += dx[:3]
        self.velocity += dx[3:6]
        self.q_i_from_b = self.q_i_from_b.boxplusr(dx[6:])

    def lmMeas(self, lm):
        z, Jr = self.q_i_from_b.inv_rotate(self.position - lm, Jr=np.eye(3))
        return z, Jr

class EKF:
    def __init__(self):
        self.Q = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]) * 1
        self.R_alt_ = 1e-2
        self.R_gps_ = np.diag([.25, .25, .5])
        self.R_lm_ = np.diag([1e-3, 1e-3, 1e-3])
        self.R_pos_ = np.diag([.01, .01, .01])
        self.R_att_ = np.diag([1e-4, 1e-4, 1e-4])

    # Direct position update like mocap
    def posUpdate(self, quad, z):
        z_hat = quad.position.copy()
        H = np.block([np.eye(3), np.zeros((3, 6))])

        r = z - z_hat
        S = H @ quad.P_ @ H.T + self.R_pos_

        K = quad.P_ @ H.T @ np.linalg.inv(S)

        dx = K @ r
        quad.boxplusr(dx)
        M = np.eye(9) - K @ H
        quad.P_ = M @ quad.P_ @ M.T + K @ self.R_pos_ @ K.T

        return quad

    def attUpdate(self, quad, z):
        z_hat = quad.q_i_from_b.R
        H = np.block([np.zeros((3, 6)), np.eye(3)])

        r = SO3.vee(z_hat @ z)
        S = H @ quad.P_ @ H.T + self.R_att_

        K = quad.P_ @ H.T @ np.linalg.inv(S)

        dx = K @ r
        quad.boxplusr(dx)
        M = np.eye(9) - K @ H
        quad.P_ = M @ quad.P_ @ M.T + K @ self.R_att_ @ K.T

        return quad

    def lmUpdate(self, quad, zs, lms):
        for z, lm in zip(zs, lms):
            z_hat, Jr = quad.lmMeas(lm)

            H = np.zeros((3,9))
            # H[:, :3] = # positions jacobians
            H[:, 6:] = Jr

            y = z - z_hat
            S = H @ quad.P_ @ H.T + self.R_lm_

            K = quad.P_ @ H.T @ np.linalg.inv(S)

            dx = K @ y
            quad.boxplusr(dx)
            quad.P_ -= K @ S @ K.T

if __name__=="__main__":
    t0 = 0.0
    # tf = 60.0
    tf = 60.0
    dt = 0.01 # IMU update rate of 100Hz

    lm_list = [np.array([10, 10, 10]), np.array([10, -10, 10]), np.array([-10, 10, 10]), np.array([-10, -10, 10])]

    params = QuadPrams(1.0)
    traj = Trajectory(params, False)
    # quad = Quadrotor(params, traj, 4.0)
    quad = Quadrotor(params, traj, 0.0)
    truth_quad = Quadrotor(params, traj)
    dr_quad = Quadrotor(params, traj)
    ekf = EKF()

    t_hist = np.arange(t0, tf, dt)

    x_hist, v_hist, euler_hist = [], [], []
    truth_x_hist, truth_v_hist, truth_euler_hist = [], [], []
    dr_x_hist, dr_v_hist, dr_euler_hist = [], [], []
    dt_rot, dt_pos, dt_gps = 0.0, 0.0, 0.0
    pos_cov, vel_cov, att_cov = [], [], []
    for t in t_hist:
        pos, v, ab, R_i_from_b, wb = traj.calcStep(t)
        eta_a = np.random.multivariate_normal(np.zeros(3), R_accel)
        eta_g = np.random.multivariate_normal(np.zeros(3), R_gyro)
        # eta_a = np.zeros(3)
        # eta_g = np.zeros(3)
        truth_quad.propogateDynamics(ab, wb, dt)
        quad.propogateDynamics(ab+eta_a, wb+eta_g, dt)
        dr_quad.propogateDynamics(ab+eta_a, wb+eta_g, dt)

        if dt_pos > params.t_pos:
            eta = np.random.multivariate_normal(np.zeros(3), ekf.R_pos_)
            z = truth_quad.position + eta
            quad = ekf.posUpdate(quad, z)
            dt_pos = 0.0

        if dt_rot > params.t_rot:
            eta = np.random.multivariate_normal(np.zeros(3), ekf.R_att_)
            z = R_i_from_b.boxplusr(eta).R
            quad = ekf.attUpdate(quad, z)
            dt_rot = 0

        # if dt_gps > params.t_gps:
            # pass

        dt_rot += dt
        dt_pos += dt
        # dt_gps += dt

        x_hist.append(quad.position.copy())
        v_hist.append(quad.velocity.copy())
        euler_hist.append(quad.q_i_from_b.euler)
        truth_x_hist.append(truth_quad.position.copy())
        truth_v_hist.append(truth_quad.velocity.copy())
        truth_euler_hist.append(truth_quad.q_i_from_b.euler)
        dr_x_hist.append(dr_quad.position.copy())
        dr_v_hist.append(dr_quad.velocity.copy())
        dr_euler_hist.append(dr_quad.q_i_from_b.euler)
        pos_cov.append(np.diag(quad.P_[:3, :3]))
        vel_cov.append(np.diag(quad.P_[3:6, 3:6]))
        att_cov.append(np.diag(quad.P_[-3:, -3:]))



    x_hist = np.array(x_hist).T
    v_hist = np.array(v_hist).T
    euler_hist = np.array(euler_hist).T
    truth_x_hist = np.array(truth_x_hist).T
    truth_v_hist = np.array(truth_v_hist).T
    truth_euler_hist = np.array(truth_euler_hist).T
    dr_x_hist = np.array(dr_x_hist).T
    dr_v_hist = np.array(dr_v_hist).T
    dr_euler_hist = np.array(dr_euler_hist).T
    pos_cov = 2 * np.sqrt(np.array(pos_cov).T)
    vel_cov = 2 * np.sqrt(np.array(vel_cov).T)
    att_cov = 2 * np.sqrt(np.array(att_cov).T)


    fig1, ax1 = plt.subplots(nrows=3, ncols=1)
    ax1[0].plot(t_hist, truth_x_hist[0], 'g', label='Truth')
    ax1[1].plot(t_hist, truth_x_hist[1], 'g')
    ax1[2].plot(t_hist, truth_x_hist[2], 'g')
    ax1[0].plot(t_hist, x_hist[0], 'b', label='Est')
    ax1[1].plot(t_hist, x_hist[1], 'b')
    ax1[2].plot(t_hist, x_hist[2], 'b')
    # ax1[0].plot(t_hist, dr_x_hist[0], label='DR')
    # ax1[1].plot(t_hist, dr_x_hist[1])
    # ax1[2].plot(t_hist, dr_x_hist[2])
    ax1[0].plot(t_hist, pos_cov[0] + truth_x_hist[0], 'r', label='Cov')
    ax1[1].plot(t_hist, pos_cov[1] + truth_x_hist[1], 'r')
    ax1[2].plot(t_hist, pos_cov[2] + truth_x_hist[2], 'r')
    ax1[0].plot(t_hist, -pos_cov[0] + truth_x_hist[0], 'r')
    ax1[1].plot(t_hist, -pos_cov[1] + truth_x_hist[1], 'r')
    ax1[2].plot(t_hist, -pos_cov[2] + truth_x_hist[2], 'r')

    ax1[0].set_title("Positions vs Time")
    ax1[0].legend()

    fig2, ax2 = plt.subplots(nrows=3, ncols=1)
    ax2[0].plot(t_hist, truth_v_hist[0], 'g', label='Truth')
    ax2[1].plot(t_hist, truth_v_hist[1], 'g')
    ax2[2].plot(t_hist, truth_v_hist[2], 'g')
    ax2[0].plot(t_hist, v_hist[0], 'b', label='Est')
    ax2[1].plot(t_hist, v_hist[1], 'b')
    ax2[2].plot(t_hist, v_hist[2], 'b')
    # ax2[0].plot(t_hist, dr_v_hist[0], label='DR')
    # ax2[1].plot(t_hist, dr_v_hist[1])
    # ax2[2].plot(t_hist, dr_v_hist[2])
    ax2[0].plot(t_hist, vel_cov[0] + truth_v_hist[0], 'r', label='Cov')
    ax2[1].plot(t_hist, vel_cov[1] + truth_v_hist[1], 'r')
    ax2[2].plot(t_hist, vel_cov[2] + truth_v_hist[2], 'r')
    ax2[0].plot(t_hist, -vel_cov[0] + truth_v_hist[0], 'r')
    ax2[1].plot(t_hist, -vel_cov[1] + truth_v_hist[1], 'r')
    ax2[2].plot(t_hist, -vel_cov[2] + truth_v_hist[2], 'r')
    ax2[0].set_title("Velocity vs Time")
    ax2[0].legend()

    # Cov not in euler space. rethink plotting
    fig3, ax3 = plt.subplots(nrows=3, ncols=1)
    ax3[0].plot(t_hist, truth_euler_hist[0], label='Truth')
    ax3[1].plot(t_hist, truth_euler_hist[1])
    ax3[2].plot(t_hist, truth_euler_hist[2])
    ax3[0].plot(t_hist, euler_hist[0], label='Est')
    ax3[1].plot(t_hist, euler_hist[1])
    ax3[2].plot(t_hist, euler_hist[2])
    # ax3[0].plot(t_hist, dr_euler_hist[0], label='DR')
    # ax3[1].plot(t_hist, dr_euler_hist[1])
    # ax3[2].plot(t_hist, dr_euler_hist[2])
    ax3[0].set_title("Euler Angles vs Time")
    ax3[0].legend()

    fig4, ax4 = plt.subplots(nrows=1, ncols=1)
    ax4.plot(truth_x_hist[0], truth_x_hist[1], label='Truth')
    ax4.plot(x_hist[0], x_hist[1], label='Est')
    # ax4.plot(dr_x_hist[0], dr_x_hist[1], label='DR')
    ax4.legend()

    plt.show()
