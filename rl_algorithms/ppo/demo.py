from arguments import get_args
from models import cnn_net, mlp_net
import torch
import cv2
import numpy as np
import gym
from rl_utils.env_wrapper.frame_stack import VecFrameStack
from rl_utils.env_wrapper.atari_wrapper import make_atari, wrap_deepmind

# denormalize
def normalize(x, mean, std, clip=10):
    x -= mean
    x /= (std + 1e-8)
    return np.clip(x, -clip, clip)

# get tensors for the agent
def get_tensors(obs, env_type, filters=None):
    if env_type == 'atari':
        tensor = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    elif env_type == 'mujoco':
        tensor = torch.tensor(normalize(obs, filters.rs.mean, filters.rs.std), dtype=torch.float32).unsqueeze(0)
    return tensor

if __name__ == '__main__':
    # get the arguments
    args = get_args()
    # create the environment
    if args.env_type == 'atari':
        env = make_atari(args.env_name)
        env = wrap_deepmind(env, frame_stack=True)
    elif args.env_type == 'mujoco':
        env = gym.make(args.env_name)
    # get the model path
    model_path = args.save_dir + args.env_name + '/model.pt'
    # create the network
    if args.env_type == 'atari':
        network = cnn_net(env.action_space.n)
        network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        filters = None
    elif args.env_type == 'mujoco':
        network = mlp_net(env.observation_space.shape[0], env.action_space.shape[0], args.dist)
        net_models, filters = torch.load(model_path, map_location=lambda storage, loc: storage)
        # load models 
        network.load_state_dict(net_models)
    # start to play the demo
    obs = env.reset()
    reward_total = 0
    # just one episode
    while True:
        env.render()
        with torch.no_grad():
            obs_tensor = get_tensors(obs, args.env_type, filters)
            _, pi = network(obs_tensor)
            # get actions
            if args.env_type == 'atari':
                actions = torch.argmax(pi, dim=1).item()
            elif args.env_type == 'mujoco':
                if args.dist == 'gauss':
                    mean, _ = pi
                    actions = mean.numpy().squeeze()
                elif args.dist == 'beta':
                    alpha, beta = pi
                    actions = (alpha - 1) / (alpha + beta - 2)
                    actions = actions.numpy().squeeze()
                    actions = -1 + 2 * actions 
        obs_, reward, done, _ = env.step(actions)
        reward_total += reward
        if done:
            break
        obs = obs_
    print('the rewrads is: {}'.format(reward_total))
