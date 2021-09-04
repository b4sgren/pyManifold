import sys
sys.path.append("..")
import numpy as np

from quaternion import Quaternion, skew
from se3 import SE3

np.set_printoptions(linewidth=200)

def inverseJacobian(q):
    dx = 1e-4
    Jr = np.zeros((3,3))
    Jl = np.zeros((3,3))
    for i in range(3):
        delta = np.zeros(3)
        delta[i] = dx

        q2 = q.boxplusr(delta)
        q2_l = q.boxplusl(delta)

        vec = q2.inv().boxminusr(q.inv())
        vecl = q2_l.inv().boxminusl(q.inv())
        Jr[:,i] = vec/dx
        Jl[:,i] = vecl/dx
    return Jr, Jl

def compositionJacobian(q, q2):
    dx = 1e-4
    Jr = np.zeros((3,3))
    Jl = np.zeros((3,3))
    Jr2 = np.zeros((3,3))
    Jl2 = np.zeros((3,3))

    for i in range(3):
        delta = np.zeros(3)
        delta[i] = dx

        qd = q.boxplusr(delta)
        q2d = q2.boxplusr(delta)

        vec1r = (qd * q2).boxminusr(q*q2)
        vec2r = (q * q2d).boxminusr(q*q2)

        Jr[:,i] = vec1r/dx
        Jr2[:,i] = vec2r/dx

        qd = q.boxplusl(delta)
        q2d = q2.boxplusl(delta)

        vec1l = (qd*q2).boxminusl(q*q2)
        vec2l = (q*q2d).boxminusl(q*q2)

        Jl[:,i] = vec1l/dx
        Jl2[:,i] = vec2l/dx
    return Jr, Jl, Jr2, Jl2

def ExponentialJacobian(theta):
    Jr = np.zeros((3,3))
    Jl = np.zeros((3,3))

    dx = 1e-4
    for i in range(3):
        theta2 = theta.copy()
        theta2[i] += dx

        q = Quaternion.Exp(theta)
        q2 = Quaternion.Exp(theta2)

        vecr = q2.boxminusr(q)
        vecl = q2.boxminusl(q)

        Jr[:,i] = vecr/dx
        Jl[:,i] = vecl/dx
    return Jr, Jl

def LogJacobian(q):
    Jr, Jl = np.zeros((3,3)), np.zeros((3,3))

    dx = 1e-4
    for i in range(3):
        delta = np.zeros(3)
        delta[i] = dx


        q2r = q.boxplusr(delta)
        q2l = q.boxplusl(delta)

        vecr = Quaternion.Log(q2r) - Quaternion.Log(q)
        vecl = Quaternion.Log(q2l) - Quaternion.Log(q)

        Jr[:,i] = vecr/dx
        Jl[:,i] = vecl/dx
    return Jr, Jl

def rotationJacobian(q, v):
    Jr, Jl = np.zeros((3,3)), np.zeros((3,3))

    dx = 1e-4
    for i in range(3):
        delta = np.zeros(3)
        delta[i] = dx

        q2r = q.boxplusr(delta)
        q2l = q.boxplusl(delta)

        vecr = q2r.rota(v) - q.rota(v)
        vecl = q2l.rota(v) - q.rota(v)

        Jr[:,i] = vecr/dx
        Jl[:,i] = vecl/dx

    return Jr, Jl

def se3InverseJacobian(T):
    Jr, Jl = np.eye(6), np.eye(6)

    dx = 1e-4
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = dx

        Tr = T.boxplusr(delta)
        Tl = T.boxplusl(delta)

        vecr = Tr.inv().boxminusr(T.inv())
        vecl = Tl.inv().boxminusl(T.inv())

        Jr[:,i] = vecr/dx
        Jl[:,i] = vecl/dx

    return Jr, Jl

def se3ComposeJacobians(T1, T2):
    Jr1, Jl1 = np.eye(6), np.eye(6)
    Jr2, Jl2 = np.eye(6), np.eye(6)

    T3 = T1.compose(T2)
    dx = 1e-4
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = dx

        T1r = T1.boxplusr(delta)
        T1l = T1.boxplusl(delta)

        vecr = (T1r.compose(T2)).boxminusr(T3)
        vecl = (T1l.compose(T2)).boxminusl(T3)

        Jr1[:,i] = vecr/dx
        Jl1[:,i] = vecl/dx

        T2r = T2.boxplusr(delta)
        T2l = T2.boxplusl(delta)

        vecr = (T1.compose(T2r)).boxminusr(T3)
        vecl = (T1.compose(T2l)).boxminusl(T3)

        Jr2[:,i] = vecr/dx
        Jl2[:,i] = vecl/dx

    return Jr1, Jl1, Jr2, Jl2

def se3ExpJacobian(tau):
    Jr, Jl = np.eye(6), np.eye(6)

    T = SE3.Exp(tau)
    dx = 1e-4
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = dx

        tau2 = tau + delta
        T2 = SE3.Exp(tau2)

        vecr = T2.boxminusr(T)
        vecl = T2.boxminusl(T)

        Jr[:,i] = vecr/dx
        Jl[:,i] = vecl/dx
    return Jr, Jl

def se3LogJacobian(T):
    Jr, Jl = np.eye(6), np.eye(6)

    tau = SE3.Log(T)
    dx = 1e-4
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = dx

        Tr = T.boxplusr(delta)
        Tl = T.boxplusl(delta)

        vecr = SE3.Log(Tr) - tau
        vecl = SE3.Log(Tl) - tau

        Jr[:,i] = vecr/dx
        Jl[:,i] = vecl/dx

    return Jr, Jl

def transJacobian(T, pt):
    Jr, Jl = np.zeros((3,6)), np.zeros((3,6))

    pt2 = T.transa(pt)
    dx = 1e-4
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = dx

        Tr = T.boxplusr(delta)
        Tl = T.boxplusl(delta)

        vecr = Tr.transa(pt) - pt2
        vecl = Tl.transa(pt) - pt2

        Jr[:,i] = vecr/dx
        Jl[:,i] = vecl/dx

    return Jr, Jl

if __name__=="__main__":
    # Quaternion Jacobians
    # Inversion is correct
    q = Quaternion.random()
    q_inv, Jr = q.inv(Jr=np.eye(3))
    q_inv, Jl = q.inv(Jl=np.eye(3))
    Jr_num, Jl_num = inverseJacobian(q)

    # Composition is correct
    q2 = Quaternion.random()
    q3, Jr1 = q.compose(q2, Jr=np.eye(3))
    q3, Jr2 = q.compose(q2, Jr2=np.eye(3))
    q3, Jl1 = q.compose(q2, Jl=np.eye(3))
    q3, Jl2 = q.compose(q2, Jl2=np.eye(3))
    Jr1n, Jl1n, Jr2n, Jl2n = compositionJacobian(q, q2)

    # Exponential is correct
    theta = np.random.uniform(-np.pi, np.pi, size=3)
    q, Jr = Quaternion.Exp(theta, Jr=np.eye(3))
    q, Jl = Quaternion.Exp(theta, Jl=np.eye(3))
    Jrn, Jln = ExponentialJacobian(theta)
    debug = 1

    # Logarithm Jacobian is correct
    theta, Jr = Quaternion.Log(q, Jr=np.eye(3))
    theta, Jl = Quaternion.Log(q, Jl=np.eye(3))
    Jrn, Jln = LogJacobian(q)

    # Rotation jacobians
    vec = np.random.uniform(-10, 10, size=3)
    v2, Jr = q.rota(vec, Jr=np.eye(3))
    v2, Jl = q.rota(vec, Jl=np.eye(3))
    Jrn, Jln = rotationJacobian(q, vec)
    debug = 1

    # Jacobian of inverse se3 works
    T = SE3.random()
    T_inv, Jr = T.inv(Jr = np.eye(6))
    T_inv, Jl = T.inv(Jl = np.eye(6))
    Jrn, Jln = se3InverseJacobian(T)

    # Jacobian of composition works
    T2 = SE3.random()
    T3, Jr1 = T.compose(T2, Jr=np.eye(6))
    T3, Jl1 = T.compose(T2, Jl=np.eye(6))
    T3, Jr2 = T.compose(T2, Jr2=np.eye(6))
    T3, Jl2 = T.compose(T2, Jl2=np.eye(6))
    Jr1n, Jl1n, Jr2n, Jl2n = se3ComposeJacobians(T, T2)

    # Exponential Jacobian works
    rho = np.random.uniform(-10, 10, size=3)
    theta = np.random.uniform(-np.pi, np.pi, size=3)
    tau = np.array([*rho, *theta])
    T, Jr = SE3.Exp(tau, Jr=np.eye(6))
    T, Jl = SE3.Exp(tau, Jl=np.eye(6))
    Jrn, Jln = se3ExpJacobian(tau)

    # Logarithm jacobian works
    tau, Jr = SE3.Log(T, Jr=np.eye(6))
    tau, Jl = SE3.Log(T, Jl=np.eye(6))
    Jrn, Jln = se3LogJacobian(T)
    debug = 1

    # Jacobians of transformation works
    pt = np.random.uniform(-10, 10, size=3)
    pt2, Jr = T.transa(pt, Jr=np.eye(6))
    pt2, Jl = T.transa(pt, Jl=np.eye(6))
    Jrn, Jln = transJacobian(T, pt)
    debug = 1
    # Jr = [T.R.T, -T.R.T @ skew(pt)]
    # Jl = [I_3, -skew(pt2)]

    # All other quat jacobians are a composition of the above jacobians
