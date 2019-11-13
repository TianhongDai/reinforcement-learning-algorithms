from arguments import get_args
from sac_agent import sac_agent
from rl_utils.seeds.seeds import set_seeds
from rl_utils.env_wrapper.create_env import create_single_env

if __name__ == '__main__':
    args = get_args()
    # build the environment
    env = create_single_env(args)
    # set the seeds
    set_seeds(args)
    # create the agent
    sac_trainer = sac_agent(env, args)
    sac_trainer.learn()
    # close the environment
    env.close()
