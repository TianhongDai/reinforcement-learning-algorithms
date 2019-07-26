import sys
import numpy as np
from models import net
from utils import linear_schedule, select_actions, reward_recorder
from rl_utils.experience_replay.experience_replay import replay_buffer
import torch
from datetime import datetime
import os
import copy

# define the dqn agent
class dqn_agent:
    def __init__(self, env, args):
        # define some important 
        self.env = env
        self.args = args 
        # define the network
        self.net = net(self.env.action_space.n, self.args.use_dueling)
        # copy the self.net as the 
        self.target_net = copy.deepcopy(self.net)
        # make sure the target net has the same weights as the network
        self.target_net.load_state_dict(self.net.state_dict())
        if self.args.cuda:
            self.net.cuda()
            self.target_net.cuda()
        # define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        # define the replay memory
        self.buffer = replay_buffer(self.args.buffer_size)
        # define the linear schedule of the exploration
        self.exploration_schedule = linear_schedule(int(self.args.total_timesteps * self.args.exploration_fraction), \
                                                    self.args.final_ratio, self.args.init_ratio)
        # create the folder to save the models
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # set the environment folder
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    # start to do the training
    def learn(self):
        # the episode reward
        episode_reward = reward_recorder()
        obs = np.array(self.env.reset())
        td_loss = 0
        for timestep in range(self.args.total_timesteps):
            explore_eps = self.exploration_schedule.get_value(timestep)
            with torch.no_grad():
                obs_tensor = self._get_tensors(obs)
                action_value = self.net(obs_tensor)
            # select actions
            action = select_actions(action_value, explore_eps)
            # excute actions
            obs_, reward, done, _ = self.env.step(action)
            obs_ = np.array(obs_)
            # tryint to append the samples
            self.buffer.add(obs, action, reward, obs_, float(done))
            obs = obs_
            # add the rewards
            episode_reward.add_rewards(reward)
            if done:
                obs = np.array(self.env.reset())
                # start new episode to store rewards
                episode_reward.start_new_episode()
            if timestep > self.args.learning_starts and timestep % self.args.train_freq == 0:
                # start to sample the samples from the replay buffer
                batch_samples = self.buffer.sample(self.args.batch_size)
                td_loss = self._update_network(batch_samples)
            if timestep > self.args.learning_starts and timestep % self.args.target_network_update_freq == 0:
                # update the target network
                self.target_net.load_state_dict(self.net.state_dict())
            if done and episode_reward.num_episodes % self.args.display_interval == 0:
                print('[{}] Frames: {}, Episode: {}, Mean: {:.3f}, Loss: {:.3f}'.format(datetime.now(), timestep, episode_reward.num_episodes, \
                        episode_reward.mean, td_loss))
                torch.save(self.net.state_dict(), self.model_path + '/model.pt')

    # update the network
    def _update_network(self, samples):
        obses, actions, rewards, obses_next, dones = samples
        # convert the data to tensor
        obses = self._get_tensors(obses)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        obses_next = self._get_tensors(obses_next)
        dones = torch.tensor(1 - dones, dtype=torch.float32).unsqueeze(-1)
        # convert into gpu
        if self.args.cuda:
            actions = actions.cuda()
            rewards = rewards.cuda()
            dones = dones.cuda()
        # calculate the target value
        with torch.no_grad():
            # if use the double network architecture
            if self.args.use_double_net:
                q_value_ = self.net(obses_next)
                action_max_idx = torch.argmax(q_value_, dim=1, keepdim=True)
                target_action_value = self.target_net(obses_next)
                target_action_max_value = target_action_value.gather(1, action_max_idx)
            else:
                target_action_value = self.target_net(obses_next)
                target_action_max_value, _ = torch.max(target_action_value, dim=1, keepdim=True)
        # target
        expected_value = rewards + self.args.gamma * target_action_max_value * dones
        # get the real q value
        action_value = self.net(obses)
        real_value = action_value.gather(1, actions)
        loss = (expected_value - real_value).pow(2).mean()
        # start to update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # get tensors
    def _get_tensors(self, obs):
        if obs.ndim == 3:
            obs = np.transpose(obs, (2, 0, 1))
            obs = np.expand_dims(obs, 0)
        elif obs.ndim == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.tensor(obs, dtype=torch.float32)
        if self.args.cuda:
            obs = obs.cuda()
        return obs
