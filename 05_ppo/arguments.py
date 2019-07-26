import argparse

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--num-workers', type=int, default=8, help='the number of workers to collect samples')
    parse.add_argument('--env-name', type=str, default='PongNoFrameskip-v4', help='the environment name')
    parse.add_argument('--batch-size', type=int, default=4, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=2.5e-4, help='learning rate of the algorithm')
    parse.add_argument('--epoch', type=int, default=4, help='the epoch during training')
    parse.add_argument('--nsteps', type=int, default=128, help='the steps to collect samples')
    parse.add_argument('--vloss-coef', type=float, default=0.5, help='the coefficient of value loss')
    parse.add_argument('--ent-coef', type=float, default=0.01, help='the entropy loss coefficient')
    parse.add_argument('--tau', type=float, default=0.95, help='gae coefficient')
    parse.add_argument('--cuda', action='store_true', help='use cuda do the training')
    parse.add_argument('--total-frames', type=int, default=20000000, help='the total frames for training')
    parse.add_argument('--dist', type=str, default='gauss', help='the distributions for sampling actions')
    parse.add_argument('--eps', type=float, default=1e-5, help='param for adam optimizer')
    parse.add_argument('--clip', type=float, default=0.1, help='the ratio clip param')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--lr-decay', action='store_true', help='if using the learning rate decay during decay')
    parse.add_argument('--max-grad-norm', type=float, default=0.5, help='grad norm')
    parse.add_argument('--display-interval', type=int, default=10, help='the interval that display log information')
    parse.add_argument('--env-type', type=str, default='atari', help='the type of the environment')
    parse.add_argument('--log-dir', type=str, default='logs', help='the folders to save the log files')

    args = parse.parse_args()

    return args
