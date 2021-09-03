import sys
sys.path.append("..")
import numpy as np

from quaternion import Quaternion, skew

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


if __name__=="__main__":
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
    # TODO: Something wrong with these
    vec = np.random.uniform(-10, 10, size=3)
    v2, Jr = q.rota(vec, Jr=np.eye(3))
    v2, Jl = q.rota(vec, Jl=np.eye(3))
    Jrn, Jln = rotationJacobian(q, vec)
    debug = 1

    # All other quat jacobians are a composition of the above jacobians
