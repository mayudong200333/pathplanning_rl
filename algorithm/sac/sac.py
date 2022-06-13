import copy

import torch

from algorithm.common.base_class import Algorithm
from algorithm.sac.network import ActorNetwork,CriticNetwork
from algorithm.common.buffer import ReplayBuffer

import torch as th
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class SAC(Algorithm):
    def __init__(self,env,device=th.device('cuda:0' if th.cuda.is_available() else 'cpu'),discount_factor=0.99,
                 lr_rate=[3e-4,3e-4],tau=5e-3,start_steps=int(1e4),alpha=0.2,reward_scale=1.0,batch_size=256,buffer_size=int(1e6)):
        super(SAC,self).__init__(device,discount_factor)
        self.env = env

        self.num_state = env.observation_space.shape
        self.num_action = env.action_space.shape

        self.actor = ActorNetwork(self.num_state, self.num_action).to(self.device)
        self.critic = CriticNetwork(self.num_state, self.num_action).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.optimzer_actor = optim.Adam(self.actor.parameters(), lr=lr_rate[0])
        self.optimzer_critic = optim.Adam(self.critic.parameters(), lr=lr_rate[1])

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, observation_space=env.observation_space,
                                          action_space=env.action_space,
                                          device=device)

        self.tau = tau
        self.start_steps = start_steps
        self.alpha = alpha
        self.batch_size = batch_size
        self.reward_scale = reward_scale

    def is_update(self,steps):
        return steps >= max(self.start_steps,self.batch_size)

    def step(self,state,t,steps):
        t += 1
        if steps <= self.start_steps:
            action = self.env.action_space.sample()
        else:
            action,_ = self.explore(state)
        next_state,reward,done,_ = self.env.step(action)
        if t == self.env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done

        self.replay_buffer.add(state, next_state, action, reward, done_masked)

        if done:
            t = 0
            next_state = self.env.reset()

        return next_state,t

    def update(self):
        transitions = self.replay_buffer.sample(batch_size=self.batch_size, env=self.env)
        state_batch,action_batch,next_state_batch,reward_batch,done_batch = transitions.obs,transitions.actions,transitions.next_obs,transitions.rewards,transitions.dones
        self.update_critic(state_batch, action_batch, reward_batch, done_batch, next_state_batch)
        self.update_actor(state_batch)
        self.update_target()

    def update_critic(self,states,actions,rewards,dones,next_states):
        curr_qs1,curr_qs2 = self.critic(states,actions)

        with torch.no_grad():
            next_actions,log_pis = self.actor.sample(next_states)
            next_qs1,next_qs2 = self.critic_target(next_states,next_actions)
            next_qs = torch.min(next_qs1,next_qs2) - self.alpha * log_pis

        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.discount_factor * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optimzer_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optimzer_critic.step()

    def update_actor(self,states):
        actions,log_pis = self.actor.sample(states)
        qs1,qs2 = self.critic(states,actions)
        loss_actor = (self.alpha * log_pis - torch.min(qs1, qs2)).mean()

        self.optimzer_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optimzer_actor.step()

    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)

    def explore(self,state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self,state):
        state = th.tensor(state, dtype=th.float, device=self.device).view(-1, *self.num_state)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]





