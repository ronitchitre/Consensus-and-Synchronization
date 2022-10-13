import numpy
import numpy as np
import constants
from scipy.spatial.transform import Rotation as Rot

pi = np.pi
n_agent = constants.n_agent
inertia = constants.inertia
inertia_inv = np.linalg.inv(inertia)


def create_random_R():
    return Rot.random().as_matrix()


def create_random_ang():
    return constants.max_ang_vel * constants.unit_vec(np.random.rand(3))


class Agent:
    def __init__(self, R, ang_vel):
        self.R = R
        self.ang_vel = ang_vel

    def dynamics(self):
        R_dot = self.R.dot(constants.hat(self.ang_vel))
        ang_vel_dot = inertia_inv.dot(np.cross(inertia.dot(self.ang_vel), self.ang_vel))
        return R_dot, ang_vel_dot






