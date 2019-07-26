import argparse

def get_args():
    parse = argparse.ArgumentParser(description='ddpg')
    parse.add_argument('--env-name', type=str, default='Pendulum-v0', help='the training environment')
    parse.add_argument('--lr-actor', type=float, default=1e-4, help='the lr of the actor')
    parse.add_argument('--lr-critic', type=float, default=1e-3, help='the lr of the critic')
    parse.add_argument('--critic-l2-reg', type=float, default=1e-2, help='the critic reg')
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parse.add_argument('--nb-epochs', type=int, default=500, help='the epochs to train the network')
    parse.add_argument('--nb-cycles', type=int, default=20)
    parse.add_argument('--nb-train', type=int, default=50, help='number to train the agent')
    parse.add_argument('--nb-rollout-steps', type=int, default=100, help='steps to collect samples')
    parse.add_argument('--nb-test-rollouts', type=int, default=10, help='the number of test')
    parse.add_argument('--batch-size', type=int, default=128, help='the batch size to update network')
    parse.add_argument('--replay-size', type=int, default=int(1e6), help='the size of the replay buffer')
    parse.add_argument('--clip-range', type=float, default=5, help='clip range of the observation')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the place save the models')
    parse.add_argument('--polyak', type=float, default=0.95, help='the expoential weighted coefficient.')
    parse.add_argument('--total-frames', type=int, default=int(1e6), help='total frames')
    parse.add_argument('--log-dir', type=str, default='logs', help='place to save log files')
    parse.add_argument('--env-type', type=str, default=None, help='environment type')
    parse.add_argument('--seed', type=int, default=123, help='random seed')
    parse.add_argument('--display-interval', type=int, default=10, help='interval to display')
    # ddpg not support gpu
    parse.add_argument('--cuda', action='store_true', help='if use GPU')

    args = parse.parse_args()
    return args
