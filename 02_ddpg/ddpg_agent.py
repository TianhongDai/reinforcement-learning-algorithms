import numpy as np
from models import actor, critic 
import torch
import os
from datetime import datetime
from mpi4py import MPI
from rl_utils.mpi_utils.normalizer import normalizer
from rl_utils.mpi_utils.utils import sync_networks, sync_grads
from rl_utils.experience_replay.experience_replay import replay_buffer
from utils import ounoise
import copy
import gym

"""
ddpg algorithms - revised baseline version

support MPI training

"""

class ddpg_agent:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        # get the dims and action max of the environment
        obs_dims = self.env.observation_space.shape[0]
        self.action_dims = self.env.action_space.shape[0]
        self.action_max = self.env.action_space.high[0]
        # define the network
        self.actor_net = actor(obs_dims, self.action_dims)
        self.critic_net = critic(obs_dims, self.action_dims)
        # sync the weights across the mpi
        sync_networks(self.actor_net)
        sync_networks(self.critic_net)
        # build the target newtork
        self.actor_target_net = copy.deepcopy(self.actor_net)
        self.critic_target_net = copy.deepcopy(self.critic_net)
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), self.args.lr_critic, weight_decay=self.args.critic_l2_reg)
        # create the replay buffer
        self.replay_buffer = replay_buffer(self.args.replay_size)
        # create the normalizer
        self.o_norm = normalizer(obs_dims, default_clip_range=self.args.clip_range)
        # create the noise generator
        self.noise_generator = ounoise(std=0.2, action_dim=self.action_dims)
        # create the dir to save models
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        # create a eval environemnt
        self.eval_env = gym.make(self.args.env_name)
        # set seeds
        self.eval_env.seed(self.args.seed * 2 + MPI.COMM_WORLD.Get_rank())

    def learn(self):
        """
        the learning part

        """
        self.actor_net.train()
        # reset the environmenr firstly
        obs = self.env.reset()
        self.noise_generator.reset()
        # get the number of epochs
        nb_epochs = self.args.total_frames // (self.args.nb_rollout_steps * self.args.nb_cycles)
        for epoch in range(nb_epochs):
            for _ in range(self.args.nb_cycles):
                # used to update the normalizer
                ep_obs = []
                for _ in range(self.args.nb_rollout_steps):
                    with torch.no_grad():
                        inputs_tensor = self._preproc_inputs(obs)
                        pi = self.actor_net(inputs_tensor)
                        action = self._select_actions(pi)
                    # feed actions into the environment
                    obs_, reward, done, _ = self.env.step(self.action_max * action)
                    # append the rollout information into the memory
                    self.replay_buffer.add(obs, action, reward, obs_, float(done))
                    ep_obs.append(obs.copy())
                    obs = obs_
                    # if done, reset the environment
                    if done:
                        obs = self.env.reset()
                        self.noise_generator.reset()
                # then start to do the update of the normalizer
                ep_obs = np.array(ep_obs)
                self.o_norm.update(ep_obs)
                self.o_norm.recompute_stats()
                # then start to update the network
                for _ in range(self.args.nb_train):
                    a_loss, c_loss = self._update_network()
                    # update the target network
                    self._soft_update_target_network(self.actor_target_net, self.actor_net)
                    self._soft_update_target_network(self.critic_target_net, self.critic_net)
            # start to do the evaluation
            success_rate = self._eval_agent()
            # convert back to normal
            self.actor_net.train()
            if epoch % self.args.display_interval == 0:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('[{}] Epoch: {} / {}, Frames: {}, Rewards: {:.3f}, Actor loss: {:.3f}, Critic Loss: {:.3f}'.format(datetime.now(), \
                            epoch, nb_epochs, (epoch+1) * self.args.nb_rollout_steps * self.args.nb_cycles, success_rate, a_loss, c_loss))
                    torch.save([self.actor_net.state_dict(), self.o_norm.mean, self.o_norm.std], self.model_path + '/model.pt')

    # functions to preprocess the image
    def _preproc_inputs(self, obs):
        obs_norm = self.o_norm.normalize(obs)
        inputs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        return inputs_tensor

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # TODO: Noise type now - only support ounoise
        # add the gaussian noise
        #action = action + np.random.normal(0, 0.1, self.action_dims)
        # add ou noise
        action = action + self.noise_generator.noise()
        action = np.clip(action, -1, 1)
        return action
    
    # update the network
    def _update_network(self):
        # sample the samples from the replay buffer
        samples = self.replay_buffer.sample(self.args.batch_size)
        obses, actions, rewards, obses_next, dones = samples
        # try to do the normalization of obses
        norm_obses = self.o_norm.normalize(obses)
        norm_obses_next = self.o_norm.normalize(obses_next)
        # transfer them into tensors
        norm_obses_tensor = torch.tensor(norm_obses, dtype=torch.float32)
        norm_obses_next_tensor = torch.tensor(norm_obses_next, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            actions_next = self.actor_target_net(norm_obses_next_tensor)
            q_next_value = self.critic_target_net(norm_obses_next_tensor, actions_next)
            target_q_value = rewards_tensor + (1 - dones_tensor) * self.args.gamma * q_next_value
        # the real q value
        real_q_value = self.critic_net(norm_obses_tensor, actions_tensor)
        critic_loss = (real_q_value - target_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_net(norm_obses_tensor)
        actor_loss = -self.critic_net(norm_obses_tensor, actions_real).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_net)
        self.actor_optim.step()
        # update the critic network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_net)
        self.critic_optim.step()
        return actor_loss.item(), critic_loss.item()
    
    # soft update the target network...
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)
    
    # do the evaluation
    def _eval_agent(self):
        self.actor_net.eval()
        total_success_rate = []
        for _ in range(self.args.nb_test_rollouts):
            per_success_rate = []
            obs = self.eval_env.reset()
            while True:
                with torch.no_grad():
                    inputs_tensor = self._preproc_inputs(obs)
                    pi = self.actor_net(inputs_tensor)
                    actions = pi.detach().cpu().numpy().squeeze()
                    if self.action_dims == 1:
                        actions = np.array([actions])
                obs_, reward, done, _ = self.eval_env.step(actions * self.action_max)
                per_success_rate.append(reward)
                obs = obs_
                if done:
                    break
            total_success_rate.append(np.sum(per_success_rate))
        local_success_rate = np.mean(total_success_rate)
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
