import numpy as np

tf = 30
h = 0.01
a = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
n_agent = 2
max_ang_vel = 1
kp = 1
kd = 1
ref_omega = 0
inertia = np.array([[10, 0, 0], [0, 20, 0], [0, 0, 30]])
ref_q = np.array([1, 0, 0])
marker = ['<', 'o', '^']


def unit_vec(v):
    if np.linalg.norm(v) != 0:
        v = v / np.linalg.norm(v)
    return v


def hat(v_u):
    return np.array([[0, -1 * v_u[2], v_u[1]],
                     [v_u[2], 0, -1 * v_u[0]],
                     [-1 * v_u[1], v_u[0], 0]])