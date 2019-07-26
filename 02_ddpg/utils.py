import numpy as np
import torch

# add ounoise here
class ounoise():
    def __init__(self, std, action_dim, mean=0, theta=0.15, dt=1e-2, x0=None):
        self.std = std
        self.mean = mean
        self.action_dim = action_dim
        self.theta = theta
        self.dt = dt
        self.x0 = x0
    
    # reset the noise
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.action_dim)
    
    # generate noise
    def noise(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
                self.std * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.x_prev = x
        return x
