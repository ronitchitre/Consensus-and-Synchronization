import numpy as np
import matplotlib.pyplot as plt
# import pygame
from scipy.integrate import odeint
import constants
from random import random

pi = np.pi

n_agent = constants.n_agent
ic_theta = np.array([2 * pi * random() for _ in range(n_agent)])
ic_omega = np.array([constants.max_ang_vel * random() for _ in range(n_agent)])
ic = np.concatenate([ic_theta, ic_omega])

time = np.linspace(0, constants.tf, int(constants.tf / constants.h))
order = 2


def dynamics(x, t):
    theta_vec = x[0:n_agent]
    omega_vec = x[n_agent:2 * n_agent]
    if order == 1:
        theta_vec_dot = control(x)
        omega_vec_dot = np.zeros_like(omega_vec)
    if order == 2:
        theta_vec_dot = omega_vec
        omega_vec_dot = control(x)
    x_dot = np.concatenate([theta_vec_dot, omega_vec_dot])

    return x_dot


def error(agent1, agent2):
    return agent1 - agent2


def grad_V(agent):
    return np.sin(agent)


def control(x):
    theta_vec = x[0:n_agent]
    omega_vec = x[n_agent:2 * n_agent]
    u = np.zeros_like(omega_vec)
    if order == 1:
        for i in range(len(theta_vec)):
            spring_like_force = 0
            for j in range(len(theta_vec)):
                error_ij = error(theta_vec[i], theta_vec[j])
                spring_like_force += constants.a * grad_V(error_ij)
            u[i] = -1 * constants.kp * spring_like_force
    if order == 2:
        for i in range(len(theta_vec)):
            spring_like_force = 0
            for j in range(len(theta_vec)):
                error_ij = error(theta_vec[i], theta_vec[j])
                spring_like_force += constants.a * grad_V(error_ij)
            u[i] = constants.ref_omega - (constants.kp * spring_like_force) - (constants.kd * omega_vec[i])

    return u


trajectory = odeint(dynamics, ic, time)
legend = []
for i in range(n_agent):
    plt.plot(time, trajectory[:, i] * (180 / pi))
    plt.title('theta')
    plt.xlabel('time')
    plt.ylabel('theta rad')
    legend.append(f"{i}th agent")

plt.legend(legend)
plt.show()

if order == 2:
    legend = []
    for i in range(n_agent):
        plt.plot(time, trajectory[:, i + n_agent])
        plt.title('angular velocity')
        plt.xlabel('time')
        plt.ylabel('angular velocity rad')
        legend.append(f"{i}th agent"),

    plt.legend(legend)
    plt.show()
