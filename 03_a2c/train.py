from arguments import get_args
from a2c_agent import a2c_agent
from rl_utils.env_wrapper.create_env import create_multiple_envs
from rl_utils.seeds.seeds import set_seeds
from a2c_agent import a2c_agent
import os

if __name__ == '__main__':
    # set signle thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # get args
    args = get_args()
    # create environments
    envs = create_multiple_envs(args)
    # set seeds
    set_seeds(args)
    # create trainer
    a2c_trainer = a2c_agent(envs, args)
    a2c_trainer.learn()
    # close the environment
    envs.close()
