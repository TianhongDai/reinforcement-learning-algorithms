from ddpg_agent import ddpg_agent
from arguments import get_args
from rl_utils.seeds.seeds import set_seeds
from rl_utils.env_wrapper.create_env import create_single_env
from mpi4py import MPI
import os

if __name__ == '__main__':
    # set thread and mpi stuff
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # train the network
    args = get_args()
    # build up the environment
    env = create_single_env(args, MPI.COMM_WORLD.Get_rank())
    # set the random seeds
    set_seeds(args, MPI.COMM_WORLD.Get_rank())
    # start traininng
    ddpg_trainer = ddpg_agent(env, args)
    ddpg_trainer.learn()
    # close the environment
    env.close()
