import numpy as np
import torch
from models import net
from datetime import datetime
from utils import select_actions, evaluate_actions, discount_with_dones
import os

class a2c_agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args
        # define the network
        self.net = net(self.envs.action_space.n)
        if self.args.cuda:
            self.net.cuda()
        # define the optimizer
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.args.lr, eps=self.args.eps, alpha=self.args.alpha)
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # check the saved path for envs..
        self.model_path = self.args.save_dir + self.args.env_name + '/'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # get the obs..
        self.batch_ob_shape = (self.args.num_workers * self.args.nsteps,) + self.envs.observation_space.shape
        self.obs = np.zeros((self.args.num_workers,) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        self.obs[:] = self.envs.reset()
        self.dones = [False for _ in range(self.args.num_workers)]

    # train the network..
    def learn(self):
        num_updates = self.args.total_frames // (self.args.num_workers * self.args.nsteps)
        # get the reward to calculate other information
        episode_rewards = np.zeros((self.args.num_workers, ), dtype=np.float32)
        final_rewards = np.zeros((self.args.num_workers, ), dtype=np.float32)
        # start to update
        for update in range(num_updates):
            mb_obs, mb_rewards, mb_actions, mb_dones = [],[],[],[]
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    input_tensor = self._get_tensors(self.obs)
                    _, pi = self.net(input_tensor)
                # select actions
                actions = select_actions(pi)
                cpu_actions = actions.squeeze(1).cpu().numpy()
                # start to store the information
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(cpu_actions)
                mb_dones.append(self.dones)
                # step
                obs, rewards, dones, _ = self.envs.step(cpu_actions)
                # start to store the rewards
                self.dones = dones
                mb_rewards.append(rewards)
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n]*0
                self.obs = obs
                episode_rewards += rewards
                # get the masks
                masks = np.array([0.0 if done else 1.0 for done in dones], dtype=np.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks
                # update the obs
            mb_dones.append(self.dones)
            # process the rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
            mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
            mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
            mb_masks = mb_dones[:, :-1]
            mb_dones = mb_dones[:, 1:]
            # calculate the last value
            with torch.no_grad():
                input_tensor = self._get_tensors(self.obs)
                last_values, _ = self.net(input_tensor)
            # compute returns
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values.detach().cpu().numpy().squeeze())):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.args.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.args.gamma)
                mb_rewards[n] = rewards
            mb_rewards = mb_rewards.flatten()
            mb_actions = mb_actions.flatten()
            # start to update network
            vl, al, ent = self._update_network(mb_obs, mb_rewards, mb_actions)
            if update % self.args.log_interval == 0:
                print('[{}] Update: {}/{}, Frames: {}, Rewards: {:.1f}, VL: {:.3f}, PL: {:.3f}, Ent: {:.2f}, Min: {}, Max:{}'.format(\
                    datetime.now(), update, num_updates, (update+1)*(self.args.num_workers * self.args.nsteps),\
                    final_rewards.mean(), vl, al, ent, final_rewards.min(), final_rewards.max()))
                torch.save(self.net.state_dict(), self.model_path + 'model.pt')
    
    # update_network
    def _update_network(self, obs, returns, actions):
        # evaluate the actions
        input_tensor = self._get_tensors(obs)
        values, pi = self.net(input_tensor)
        # define the tensor of actions, returns
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        if self.args.cuda:
            returns = returns.cuda()
            actions = actions.cuda()
        # evaluate actions
        action_log_probs, dist_entropy = evaluate_actions(pi, actions)
        # calculate advantages...
        advantages = returns - values
        # get the value loss
        value_loss = advantages.pow(2).mean()
        # get the action loss
        action_loss = -(advantages.detach() * action_log_probs).mean()
        # total loss
        total_loss = action_loss + self.args.value_loss_coef * value_loss - self.args.entropy_coef * dist_entropy
        # start to update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        return value_loss.item(), action_loss.item(), dist_entropy.item()
    
    # get the tensors...
    def _get_tensors(self, obs):
        input_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)
        if self.args.cuda:
            input_tensor = input_tensor.cuda()
        return input_tensor
