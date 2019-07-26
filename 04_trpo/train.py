from arguments import get_args
from rl_utils.seeds.seeds import set_seeds
from rl_utils.env_wrapper.create_env import create_single_env
from trpo_agent import trpo_agent

if __name__ == '__main__':
    args = get_args()
    # make environemnts
    env = create_single_env(args)
    # set the random seeds
    set_seeds(args)
    # create trpo trainer
    trpo_trainer = trpo_agent(env, args)
    trpo_trainer.learn()
    # close the environment
    env.close()
