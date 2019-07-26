from arguments import get_args
from models import net
import torch
from utils import select_actions
import cv2
import numpy as np
from rl_utils.env_wrapper.frame_stack import VecFrameStack
from rl_utils.env_wrapper.atari_wrapper import make_atari, wrap_deepmind

# update the current observation
def get_tensors(obs):
    input_tensor = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    return input_tensor

if __name__ == "__main__":
    args = get_args()
    # create environment
    #env = VecFrameStack(wrap_deepmind(make_atari(args.env_name)), 4)
    env = make_atari(args.env_name)
    env = wrap_deepmind(env, frame_stack=True)
    # get the model path
    model_path = args.save_dir + args.env_name + '/model.pt'
    network = net(env.action_space.n)
    network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage)) 
    obs = env.reset()
    while True:
        env.render()
        # get the obs
        with torch.no_grad():
            input_tensor = get_tensors(obs)
            _, pi = network(input_tensor)
        actions = select_actions(pi, True)
        obs, reward, done, _ = env.step([actions])
    env.close()
