import numpy as np
import matplotlib.pyplot as plt
import constants
import body
from solver_rk4method import rk4method
from mpl_toolkits import mplot3d

pi = np.pi
n_agent = constants.n_agent
inertia = constants.inertia
inertia_inv = np.linalg.inv(inertia)


def create_ic():
    ic = np.empty(shape=2 * n_agent, dtype=object)
    for i in range(n_agent):
        ic[i] = body.create_random_R()
        ic[i + n_agent] = body.create_random_ang()

    return ic


def error(agent_i, agent_j):
    return agent_j.T.dot(agent_i)


def grad_V(agent):
    return (agent - agent.T) / 2


def control(y):
    u_i_array = np.empty(shape=n_agent, dtype=object)
    for i in range(n_agent):
        spring_like_force = np.zeros((3, 3), dtype=float)
        omega_hat = constants.hat(y[i + n_agent])
        for j in range(n_agent):
            error_ij = error(y[i], y[j])
            spring_like_force += grad_V(error_ij)
        u_i = (-1 * constants.kp * spring_like_force) - (constants.kd * omega_hat)
        u_i_array[i] = constants.hat_inv(u_i) + constants.ref_omega
    return u_i_array


def func_ode(t, y):
    y_dot = np.empty(shape=2 * n_agent, dtype=object)
    control_app = control(y)
    for i in range(n_agent):
        agent = body.Agent(y[i], y[i + n_agent])
        R_dot, ang_vel_dot = agent.dynamics()
        y_dot[i] = R_dot
        y_dot[i + n_agent] = ang_vel_dot + control_app[i]
    return y_dot


def plots(trajectory, time):
    arrow_pos = []
    for i in range(n_agent):
        q_array = np.zeros((trajectory.shape[0], 3))
        for state in range(trajectory.shape[0]):
            q_array[state] = trajectory[state][i].dot(constants.ref_q)
        arrow_pos.append(q_array)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Data for a sphere
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, alpha=0.3)

    for data in arrow_pos:
        ax.scatter3D(data[:, 0], data[:, 1], data[:, 2])
    plt.show()

    ang_vel_arr = np.zeros((time.shape[0], n_agent * 3))
    for i in range(time.shape[0]):
        for j in range(n_agent):
            ang_vel_arr[i][3 * j:3 * j + 3] = trajectory[i][n_agent + j]

    for i in range(n_agent):
        plt.plot(time, ang_vel_arr[:, i])
        plt.plot(time, ang_vel_arr[:, i + 1])
        plt.plot(time, ang_vel_arr[:, i + 2])
        plt.title(f'angular velocity {i}th agent')
        plt.xlabel('time')
        plt.ylabel('angular velocity rad')
        plt.legend(['x', 'y', 'z'])
        plt.show()


ic = create_ic()
time = np.linspace(0, constants.tf, int(constants.tf / constants.h))
trajectory = rk4method(func_ode, ic, time, 2 * n_agent)
print('ode solved')
plots(trajectory, time)

for agent_i in range(n_agent):
    for agent_j in range(n_agent):
        if agent_i == agent_j:
            continue
        error_ij = error(trajectory[-1][agent_i], trajectory[-1][agent_j])
        print(f'error between {agent_i}th and {agent_j}th agent - {error_ij}')

for ang_vel in trajectory[-1][n_agent:2 * n_agent]:
    print(f'final angular velocity{ang_vel}')
