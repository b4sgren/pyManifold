import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from so3 import SO3

class QuadPrams:
    def __init__(self, mass):
        self.mass = mass
        self.J = np.eye(3) # Update this

        self.t_alt = .25
        self.t_pos = .25
        self.t_gps = 1.0

# Choose between circle or Figure 8 trajectory
# Based on differential flatness in Minimum Snap Trajectory generation and control for quadrotors
class Trajectory:
    def __init__(self, quad_params, circle=False):
        self._is_circle = circle
        self._quad_params = quad_params

    def calcStep(self, t):
        w = np.pi/15.0
        g = 9.81
        if self._is_circle:
            # Complete a circle in 30s
            # Flat inputs
            x = 3 * np.cos(w*t)
            y = 3 * np.sin(w*t)
            z = -5.0
            psi = 0.0

            # World frames
            vx = -3 * w * np.sin(w*t)
            vy = 3 * w * np.cos(w*t)
            vz = 0

            # Accelerations in the world frame
            ax = -3 * w**2 * np.cos(w*t)
            ay = -3 * w**2 * np.sin(w*t)
            az = 0

            # Jerk
            jx = 3 * w**3 * np.sin(w*t)
            jy = -3 * w**3 * np.cos(w*t)
            jz = 0
            ad = np.array([jx, jy, jz])

            # Get rotation
            t = np.array([ax, ay, az + g])
            R_i_from_b = self.getRotation(t, psi)

            # Get angular rates
            u1 = self._quad_params.mass * np.linalg.norm(t)
            zB = R_i_from_b[:,-1]
            hw = self._quad_params.mass/u1 * (ad - (zB @ ad)*zB)
            p = -hw @ R_i_from_b[:,1]
            q = hw @ R_i_from_b[:,0]
            r = 0 # Because psi_dot is 0

        else:
            # Complete Figure 8 in 30s
            x = 3 * np.cos(w*t)
            y =  3 * np.cos(w*t) * np.sin(w*t)
            z = -5.0
            psi = 0.0

            vx = -3 * w * np.sin(w*t)
            vy = -3 * w  * (np.cos(w*t)**2 - np.sin(w*t)**2)
            vz = 0.0

            ax = -3 * w**2 * np.cos(w*t)
            ay = 12 * w**2 * np.cos(w*t) * np.sin(w*t)
            az = 0.0

            jx = 3 * w**3 * np.sin(w*t)
            jy = 12 * w**3 * (np.cos(w*t)**2 - np.sin(w*t)**2)
            jz = 0
            ad = np.array([jx, jy, jz])

            t = np.array([ax, ay, az+g])
            R_i_from_b = self.getRotation(t, psi)

            # Get angular rates
            u1 = self._quad_params.mass * np.linalg.norm(t)
            zB = R_i_from_b[:,-1]
            hw = self._quad_params.mass/u1 * (ad - (zB @ ad)*zB)
            p = -hw @ R_i_from_b[:,1]
            q = hw @ R_i_from_b[:,0]
            r = 0 # Because psi_dot is 0

        # return np.array([x, y, z]), np.array([vx, vy, vz]), R_i_from_b.T@np.array([ax, ay, az+g]), SO3(R_i_from_b), np.array([p, q, r])
        return np.array([x, y, z]), R_i_from_b.T@np.array([vx, vy, vz]), R_i_from_b.T@np.array([ax, ay, az+g]), SO3(R_i_from_b), np.array([p, q, r])

    def getRotation(self,t, psi):
        zB = t / np.linalg.norm(t)
        xC = np.array([np.cos(psi), np.sin(psi), 0])
        yB = np.cross(zB, xC)
        yB /= np.linalg.norm(yB)
        xB = np.cross(yB, zB)

        return np.vstack((xB, yB, zB)).T

if __name__=="__main__":
    params = QuadPrams(1.0)
    traj = Trajectory(params, True)
    # traj = Trajectory(params, False)

    t = np.linspace(0, 60, 1000)

    x_hist, v_hist, euler_hist = [], [], []
    for i in range(1000):
        pos, v, ab, R_i_from_b, wb = traj.calcStep(t[i])
        x_hist.append(pos)
        v_hist.append(v)
        euler_hist.append(R_i_from_b.euler)

    x_hist = np.array(x_hist).T
    v_hist = np.array(v_hist).T
    euler_hist = np.array(euler_hist).T

    fig1, ax1 = plt.subplots(nrows=3, ncols=1)
    ax1[0].plot(t, x_hist[0])
    ax1[1].plot(t, x_hist[1])
    ax1[2].plot(t, x_hist[2])
    ax1[0].set_title("Positions vs Time")

    fig2, ax2 = plt.subplots(nrows=3, ncols=1)
    ax2[0].plot(t, v_hist[0])
    ax2[1].plot(t, v_hist[1])
    ax2[2].plot(t, v_hist[2])
    ax2[0].set_title("Velocity vs Time")

    fig3, ax3 = plt.subplots(nrows=3, ncols=1)
    ax3[0].plot(t, euler_hist[0])
    ax3[1].plot(t, euler_hist[1])
    ax3[2].plot(t, euler_hist[2])
    ax3[0].set_title("Euler Angles vs Time")

    fig4, ax4 = plt.subplots(nrows=1, ncols=1)
    ax4.plot(x_hist[0], x_hist[1])

    plt.show()
