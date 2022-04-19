# imports
import copy
import time

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from scipy.signal import cont2discrete, tf2ss

# initial setup
rms = lambda data: np.sqrt(np.mean(data ** 2))
percent_rms_err = lambda estimate, actual: 100 * rms(estimate - actual) / (np.max(actual) - np.min(actual))


def plot_one_sd_blob(means, covs, ax, **kwargs):
    pearson = covs[0][1] / np.sqrt(covs[0][0] * covs[1][1])
    r_x, r_y = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=r_x * 2,height=r_y * 2, facecolor='none', **kwargs)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(np.sqrt(covs[0][0]), np.sqrt(covs[1][1])) \
        .translate(means[0], means[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def make_sinusoid(times, dt, q_pos = 0.05, q_vel = 0.05):
    amplitude = np.random.uniform(5, 10)
    frequency = np.round(np.random.uniform(1, 10), 1)
    phase = np.random.uniform(0, 2 * np.pi)
    ideal_vel = amplitude * 2*np.pi*frequency * np.cos(2*np.pi*frequency*times + phase)
    vel = ideal_vel + np.insert(np.diff(ideal_vel), 0, 0) + np.random.normal(0, np.sqrt(q_vel), times.size)
    pos = np.insert(np.cumsum(vel * dt)[:-1], 0, 0) + np.random.normal(0, np.sqrt(q_pos), times.size)
    return pos - np.mean(pos)


class AbstractKFilter:

    def measure(self):
        return self.C @ self.state

    def simulate(self, measurements):
        # 'measurements' is an m x t array, where t is the number of timesteps
        states = np.zeros((measurements.size // self.m, self.s))
        k = 0
        k_ss = 0
        for m in measurements:
            self.predict()
            self.update(m)
            states[k] = self.state
            k += 1
        return states.T

    def run_till_ss(self):
        state_init = copy.deepcopy(self.state)
        i = 0
        while not self.steady_state:
            i += 1
            self.predict()
            self.update(np.zeros(self.m,))

        self.state = state_init
        return i

    # def reset(self):
    #     self.steady_state = False
    #     self.state = np.zeros(self.s)
    #     self.P = np.zeros((self.s, self.s))
