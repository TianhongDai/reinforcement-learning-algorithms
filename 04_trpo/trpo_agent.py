import torch
import numpy as np
import os
from models import network
from rl_utils.running_filter.running_filter import ZFilter
from utils import select_actions, eval_actions, conjugated_gradient, line_search, set_flat_params_to
from datetime import datetime

class trpo_agent:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        # define the network
        self.net = network(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.old_net = network(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        # make sure the net and old net have the same parameters
        self.old_net.load_state_dict(self.net.state_dict())
        # define the optimizer
        self.optimizer = torch.optim.Adam(self.net.critic.parameters(), lr=self.args.lr)
        # define the running mean filter
        self.running_state = ZFilter((self.env.observation_space.shape[0],), clip=5)
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = self.args.save_dir + self.args.env_name + '/'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    def learn(self):
        num_updates = self.args.total_timesteps // self.args.nsteps
        obs = self.running_state(self.env.reset())
        final_reward = 0
        episode_reward = 0
        self.dones = False
        for update in range(num_updates):
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    obs_tensor = self._get_tensors(obs)
                    value, pi = self.net(obs_tensor)
                # select actions
                actions = select_actions(pi)
                # store informations
                mb_obs.append(np.copy(obs))
                mb_actions.append(actions)
                mb_dones.append(self.dones)
                mb_values.append(value.detach().numpy().squeeze())
                # start to execute actions in the environment
                obs_, reward, done, _ = self.env.step(actions)
                self.dones = done
                mb_rewards.append(reward)
                if done:
                    obs_ = self.env.reset()
                obs = self.running_state(obs_)
                episode_reward += reward
                mask = 0.0 if done else 1.0
                final_reward *= mask
                final_reward += (1 - mask) * episode_reward
                episode_reward *= mask
            # to process the rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            # compute the last state value
            with torch.no_grad():
                obs_tensor = self._get_tensors(obs)
                last_value, _ = self.net(obs_tensor)
                last_value = last_value.detach().numpy().squeeze()
            # compute the advantages
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(self.args.nsteps)):
                if t == self.args.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - mb_dones[t + 1]
                    nextvalues = mb_values[t + 1]
                delta = mb_rewards[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam
            mb_returns = mb_advs + mb_values
            # normalize the advantages
            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-5)
            # before the update, make the old network has the parameter of the current network
            self.old_net.load_state_dict(self.net.state_dict())
            # start to update the network
            policy_loss, value_loss = self._update_network(mb_obs, mb_actions, mb_returns, mb_advs)
            torch.save([self.net.state_dict(), self.running_state], self.model_path + 'model.pt')
            print('[{}] Update: {} / {}, Frames: {}, Reward: {:.3f}, VL: {:.3f}, PL: {:.3f}'.format(datetime.now(), update, \
                    num_updates, (update + 1)*self.args.nsteps, final_reward, value_loss, policy_loss))

    # start to update network
    def _update_network(self, mb_obs, mb_actions, mb_returns, mb_advs):
        mb_obs_tensor = torch.tensor(mb_obs, dtype=torch.float32)
        mb_actions_tensor = torch.tensor(mb_actions, dtype=torch.float32)
        mb_returns_tensor = torch.tensor(mb_returns, dtype=torch.float32).unsqueeze(1)
        mb_advs_tensor = torch.tensor(mb_advs, dtype=torch.float32).unsqueeze(1)
        # try to get the old policy and current policy
        values, _ = self.net(mb_obs_tensor)
        with torch.no_grad():
            _, pi_old = self.old_net(mb_obs_tensor)
        # get the surr loss
        surr_loss = self._get_surrogate_loss(mb_obs_tensor, mb_advs_tensor, mb_actions_tensor, pi_old)
        # comupte the surrogate gardient -> g, Ax = g, where A is the fisher information matrix
        surr_grad = torch.autograd.grad(surr_loss, self.net.actor.parameters())
        flat_surr_grad = torch.cat([grad.view(-1) for grad in surr_grad]).data
        # use the conjugated gradient to calculate the scaled direction vector (natural gradient)
        nature_grad = conjugated_gradient(self._fisher_vector_product, -flat_surr_grad, 10, mb_obs_tensor, pi_old)
        # calculate the scaleing ratio
        non_scale_kl = 0.5 * (nature_grad * self._fisher_vector_product(nature_grad, mb_obs_tensor, pi_old)).sum(0, keepdim=True)
        scale_ratio = torch.sqrt(non_scale_kl / self.args.max_kl)
        final_nature_grad = nature_grad / scale_ratio[0]
        # calculate the expected improvement rate...
        expected_improve = (-flat_surr_grad * nature_grad).sum(0, keepdim=True) / scale_ratio[0]
        # get the flat param ...
        prev_params = torch.cat([param.data.view(-1) for param in self.net.actor.parameters()])
        # start to do the line search
        success, new_params = line_search(self.net.actor, self._get_surrogate_loss, prev_params, final_nature_grad, \
                                expected_improve, mb_obs_tensor, mb_advs_tensor, mb_actions_tensor, pi_old)
        set_flat_params_to(self.net.actor, new_params)
        # then trying to update the critic network
        inds = np.arange(mb_obs.shape[0])
        for _ in range(self.args.vf_itrs):
            np.random.shuffle(inds)
            for start in range(0, mb_obs.shape[0], self.args.batch_size):
                end = start + self.args.batch_size
                mbinds = inds[start:end]
                mini_obs = mb_obs[mbinds]
                mini_returns = mb_returns[mbinds]
                # put things in the tensor
                mini_obs = torch.tensor(mini_obs, dtype=torch.float32)
                mini_returns = torch.tensor(mini_returns, dtype=torch.float32).unsqueeze(1)
                values, _ = self.net(mini_obs)
                v_loss = (mini_returns - values).pow(2).mean()
                self.optimizer.zero_grad()
                v_loss.backward()
                self.optimizer.step()
        return surr_loss.item(), v_loss.item()

    # get the surrogate loss
    def _get_surrogate_loss(self, obs, adv, actions, pi_old):
        _, pi = self.net(obs)
        log_prob = eval_actions(pi, actions)
        old_log_prob = eval_actions(pi_old, actions).detach()
        surr_loss = -torch.exp(log_prob - old_log_prob) * adv
        return surr_loss.mean()

    # the product of the fisher informaiton matrix and the nature gradient -> Ax
    def _fisher_vector_product(self, v, obs, pi_old):
        kl = self._get_kl(obs, pi_old)
        kl = kl.mean()
        # start to calculate the second order gradient of the KL
        kl_grads = torch.autograd.grad(kl, self.net.actor.parameters(), create_graph=True)
        flat_kl_grads = torch.cat([grad.view(-1) for grad in kl_grads])
        kl_v = (flat_kl_grads * torch.autograd.Variable(v)).sum()
        kl_second_grads = torch.autograd.grad(kl_v, self.net.actor.parameters())
        flat_kl_second_grads = torch.cat([grad.contiguous().view(-1) for grad in kl_second_grads]).data
        flat_kl_second_grads = flat_kl_second_grads + self.args.damping * v
        return flat_kl_second_grads

    # get the kl divergence between two distributions
    def _get_kl(self, obs, pi_old):
        mean_old, std_old = pi_old
        _, pi = self.net(obs)
        mean, std = pi
        # start to calculate the kl-divergence
        kl = -torch.log(std / std_old) + (std.pow(2) + (mean - mean_old).pow(2)) / (2 * std_old.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
 
    # get the tensors
    def _get_tensors(self, obs):
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
