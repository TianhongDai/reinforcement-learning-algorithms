from arguments import get_args
import gym
import torch
import numpy as np
from models import tanh_gaussian_actor

if __name__ == '__main__':
    args = get_args()
    env = gym.make(args.env_name)
    # get environment infos
    obs_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]
    action_max = env.action_space.high[0]
    # define the network
    actor_net = tanh_gaussian_actor(obs_dims, action_dims, args.hidden_size, args.log_std_min, args.log_std_max)
    # load models
    model_path = args.save_dir + args.env_name + '/model.pt'
    # load the network weights
    actor_net.load_state_dict(torch.load(model_path, map_location='cpu'))
    for ep in range(5):
        obs = env.reset()
        reward_sum = 0
        # set the maximum timesteps here...
        for _ in range(1000):
            env.render()
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                mean, std = actor_net(obs_tensor)
                actions = torch.tanh(mean).detach().numpy().squeeze()
                if action_dims == 1:
                    actions = np.array([actions])
            obs_, reward, done, _ = env.step(action_max * actions)
            reward_sum += reward
            if done:
                break
            obs = obs_
        print('the episode is: {}, the reward is: {}'.format(ep, reward_sum))
    env.close()
