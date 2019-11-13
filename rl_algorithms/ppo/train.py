from arguments import get_args
from ppo_agent import ppo_agent
from rl_utils.env_wrapper.create_env import create_multiple_envs, create_single_env
from rl_utils.seeds.seeds import set_seeds
import os

if __name__ == '__main__':
    # set signle thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # get arguments
    args = get_args()
    # start to create the environment
    if args.env_type == 'atari':
        envs = create_multiple_envs(args)
    elif args.env_type == 'mujoco':
        envs = create_single_env(args)
    else:
        raise NotImplementedError
    # create trainer
    ppo_trainer = ppo_agent(envs, args)
    # start to learn
    ppo_trainer.learn()
    # close the environment
    envs.close()
