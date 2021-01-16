import argparse

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--env-name', type=str, default='PongNoFrameskip-v4', help='the environment name')
    parse.add_argument('--batch-size', type=int, default=32, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=1e-4, help='learning rate of the algorithm')
    parse.add_argument('--buffer-size', type=int, default=10000, help='the size of the buffer')
    parse.add_argument('--cuda', action='store_true', help='if use the gpu')
    parse.add_argument('--init-ratio', type=float, default=1, help='the initial exploration ratio')
    parse.add_argument('--exploration_fraction', type=float, default=0.1, help='decide how many steps to do the exploration')
    parse.add_argument('--final-ratio', type=float, default=0.01, help='the final exploration ratio')
    parse.add_argument('--grad-norm-clipping', type=float, default=10, help='the gradient clipping')
    parse.add_argument('--total-timesteps', type=int, default=int(1e7), help='the total timesteps to train network')
    parse.add_argument('--learning-starts', type=int, default=10000, help='the frames start to learn')
    parse.add_argument('--train-freq', type=int, default=4, help='the frequency to update the network')
    parse.add_argument('--target-network-update-freq', type=int, default=1000, help='the frequency to update the target network')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--display-interval', type=int, default=10, help='the display interval')
    parse.add_argument('--env-type', type=str, default='atari', help='the environment type')
    parse.add_argument('--log-dir', type=str, default='logs', help='dir to save log information')
    parse.add_argument('--use-double-net', action='store_true', help='use double dqn to train the agent')
    parse.add_argument('--use-dueling', action='store_true', help='use dueling to train the agent')

    args = parse.parse_args()

    return args
