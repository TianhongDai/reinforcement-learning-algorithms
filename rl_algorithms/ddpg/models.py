import torch
import torch.nn as nn
import torch.nn.functional as F

# define the actor network
class actor(nn.Module):
    def __init__(self, obs_dims, action_dims):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(obs_dims, 400)
        self.fc2 = nn.Linear(400, 300)
        self.action_out = nn.Linear(300, action_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.action_out(x))
        return actions

class critic(nn.Module):
    def __init__(self, obs_dims, action_dims):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(obs_dims, 400)
        self.fc2 = nn.Linear(400 + action_dims, 300)
        self.q_out = nn.Linear(300, 1)

    def forward(self, x, actions):
        x = F.relu(self.fc1(x))
        x = torch.cat([x, actions], dim=1)
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value
