import torch
from torch import nn
from torch.nn import functional as F

class network(nn.Module):
    def __init__(self, num_states, num_actions):
        super(network, self).__init__()
        # define the critic
        self.critic = critic(num_states)
        self.actor = actor(num_states, num_actions)

    def forward(self, x):
        state_value = self.critic(x)
        pi = self.actor(x)
        return state_value, pi

class critic(nn.Module):
    def __init__(self, num_states):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        value = self.value(x)
        return value

class actor(nn.Module):
    def __init__(self, num_states, num_actions):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_mean = nn.Linear(64, num_actions)
        self.sigma_log = nn.Parameter(torch.zeros(1, num_actions))

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mean = self.action_mean(x)
        sigma_log = self.sigma_log.expand_as(mean)
        sigma = torch.exp(sigma_log)
        pi = (mean, sigma)
        
        return pi
