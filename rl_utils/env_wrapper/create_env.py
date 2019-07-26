from rl_utils.env_wrapper.atari_wrapper import make_atari, wrap_deepmind
from rl_utils.env_wrapper.multi_envs_wrapper import SubprocVecEnv
from rl_utils.env_wrapper.frame_stack import VecFrameStack
from rl_utils.logger import logger, bench
import os
import gym

"""
this functions is to create the environments

"""

def create_single_env(args, rank=0):
    # setup the log files
    if rank == 0:
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        log_path = args.log_dir + '/{}/'.format(args.env_name)
        logger.configure(log_path)
    # start to create environment
    if args.env_type == 'atari':
        # create the environment
        env = make_atari(args.env_name)
        # the monitor
        env = bench.Monitor(env, logger.get_dir())
        # use the deepmind environment wrapper
        env = wrap_deepmind(env, frame_stack=True)
    else:
        env = gym.make(args.env_name)
        # add log information
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    # set seeds to the environment to make sure the reproducebility
    env.seed(args.seed + rank)
    return env

# create multiple environments - for multiple
def create_multiple_envs(args):
    # now only support the atari games
    if args.env_type == 'atari':
        def make_env(rank):
            def _thunk():
                if not os.path.exists(args.log_dir):
                    os.mkdir(args.log_dir)
                log_path = args.log_dir + '/{}/'.format(args.env_name)
                logger.configure(log_path)
                env = make_atari(args.env_name)
                # set the seed for the environment
                env.seed(args.seed + rank)
                # set loggler
                env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
                # use the deepmind environment wrapper
                env = wrap_deepmind(env)
                return env
            return _thunk
            # put into sub processing 
        envs = SubprocVecEnv([make_env(i) for i in range(args.num_workers)])
        # then, frame stack
        envs = VecFrameStack(envs, 4)
    else:
        raise NotImplementedError
    return envs

