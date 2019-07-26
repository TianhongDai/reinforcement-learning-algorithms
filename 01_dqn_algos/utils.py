import numpy as np
import random

# linear exploration schedule
class linear_schedule:
    def __init__(self, total_timesteps, final_ratio, init_ratio=1.0):
        self.total_timesteps = total_timesteps
        self.final_ratio = final_ratio
        self.init_ratio = init_ratio

    def get_value(self, timestep):
        frac = min(float(timestep) / self.total_timesteps, 1.0)
        return self.init_ratio - frac * (self.init_ratio - self.final_ratio)

# select actions
def select_actions(action_value, explore_eps):
    action_value = action_value.cpu().numpy().squeeze()
    # select actions
    action = np.argmax(action_value) if random.random() > explore_eps else np.random.randint(action_value.shape[0])
    return action

# record the reward info of the dqn experiments
class reward_recorder:
    def __init__(self, history_length=100):
        self.history_length = history_length
        # the empty buffer to store rewards 
        self.buffer = [0.0]
        self._episode_length = 1
    
    # add rewards
    def add_rewards(self, reward):
        self.buffer[-1] += reward

    # start new episode
    def start_new_episode(self):
        if self.get_length >= self.history_length:
            self.buffer.pop(0)
        # append new one
        self.buffer.append(0.0)
        self._episode_length += 1

    # get length of buffer
    @property
    def get_length(self):
        return len(self.buffer)
    
    @property
    def mean(self):
        return np.mean(self.buffer)
    
    # get the length of total episodes
    @property 
    def num_episodes(self):
        return self._episode_length
