from arguments import get_args
import gym
from models import actor
import torch
import numpy as np

def normalize(obs, mean, std, clip):
    return np.clip((obs - mean) / std, -clip, clip)

if __name__ == '__main__':
    args = get_args()
    env = gym.make(args.env_name)
    # get environment infos
    obs_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]
    action_max = env.action_space.high[0]
    # define the network
    actor_net = actor(obs_dims, action_dims)
    # load models
    model_path = args.save_dir + args.env_name + '/model.pt'
    model, mean, std = torch.load(model_path, map_location=lambda storage, loc: storage)
    # load models into the network
    actor_net.load_state_dict(model)
    for ep in range(10):
        obs = env.reset()
        reward_sum = 0
        while True:
            env.render()
            with torch.no_grad():
                norm_obs = normalize(obs, mean, std, args.clip_range)
                norm_obs_tensor = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0)
                actions = actor_net(norm_obs_tensor)
                actions = actions.detach().numpy().squeeze()
                if action_dims == 1:
                    actions = np.array([actions])
            obs_, reward, done, _ = env.step(action_max * actions)
            reward_sum += reward
            if done:
                break
            obs = obs_
        print('the episode is: {}, the reward is: {}'.format(ep, reward_sum))
    env.close()
