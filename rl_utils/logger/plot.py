import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from rl_utils.bench import load_results

sns.set(style="dark")
sns.set_context("poster", font_scale=2, rc={"lines.linewidth": 2})
sns.set(rc={"figure.figsize": (15, 8)})
colors = sns.color_palette(palette='muted')


X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 150
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def ts2xy(ts, xaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
        y = ts.r.values
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
        y = ts.r.values
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
        y = ts.r.values
    else:
        raise NotImplementedError
    return x, y

def plot_curves(xy_list, xaxis, title, plt_order, beta=False):
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    if beta == 'dqn':
        label = ['DQN']
    elif beta == 'ddqn':
        label = ['Double-DQN']
    elif beta == 'dueling':
        label = ['Dueling-DQN']
    psub = plt.subplot(plt_order)
    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    for (i, (x, y)) in enumerate(xy_list):
        #plt.scatter(x, y, s=2)
        x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
        psub.plot(x, y_mean, label=label[i])
    psub.set_xlim([minx, maxx])
    psub.set_title(title)
    psub.legend(loc='best')
    psub.set_xlabel(xaxis)
    psub.set_ylabel("rewards")

def plot_results(dirs, num_timesteps, xaxis, task_name, plt_order, beta=False):
    tslist = []
    for dir in dirs:
        ts = load_results(dir)
        ts = ts[ts.l.cumsum() <= num_timesteps]
        tslist.append(ts)
    xy_list = [ts2xy(ts, xaxis) for ts in tslist]
    plot_curves(xy_list, xaxis, task_name, plt_order, beta)

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirs', help='List of log directories', nargs = '*', default='logs_dqn/')
    parser.add_argument('--num_timesteps', type=int, default=int(2e7))
    parser.add_argument('--xaxis', help = 'Varible on X-axis', default = X_TIMESTEPS)
    parser.add_argument('--task_name', help = 'Title of plot', default = 'BreakoutNoFrameskip-v4')
    args = parser.parse_args()
    env_name = ['BankHeistNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'KangarooNoFrameskip-v4', \
                'PongNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']
    dirs = [os.path.abspath(args.dirs + name) for name in env_name] 
    for idx in range(len(dirs)):
        plot_results([dirs[idx]], args.num_timesteps, args.xaxis, env_name[idx], 231+idx, beta='dqn')
    double_dirs = [os.path.abspath('logs_ddqn/' + name) for name in env_name]
    for idx in range(len(dirs)):
        plot_results([double_dirs[idx]], args.num_timesteps, args.xaxis, env_name[idx], 231+idx, beta='ddqn')
    dueling_dirs = [os.path.abspath('logs/' + name) for name in env_name] 
    for idx in range(len(dirs)):
        plot_results([dueling_dirs[idx]], args.num_timesteps, args.xaxis, env_name[idx], 231+idx, beta='dueling')
    plt.savefig("dueling.png")

if __name__ == '__main__':
    main()

