import torch
import numpy as np
from torch.distributions.categorical import Categorical

# select - actions
def select_actions(pi, deterministic=False):
    cate_dist = Categorical(pi)
    if deterministic:
        return torch.argmax(pi, dim=1).item()
    else:
        return cate_dist.sample().unsqueeze(-1)

# get the action log prob and entropy...
def evaluate_actions(pi, actions):
    cate_dist = Categorical(pi)
    return cate_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1), cate_dist.entropy().mean()

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.-done)
        discounted.append(r)
    return discounted[::-1]
