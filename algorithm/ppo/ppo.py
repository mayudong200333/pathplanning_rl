import copy

import torch

from algorithm.common.base_class import Algorithm
from algorithm.common.noise import OrnsteinUhlenbeckProcess
from algorithm.ppo.network import ActorNetwork,CriticNetwork
from algorithm.common.buffer import RolloutBuffer

import torch as th
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class PPO(Algorithm):
    def __init__(self,env,device=th.device('cuda:0' if th.cuda.is_available() else 'cpu'),discount_factor=0.99,
                 lr_rate=[3e-4,3e-4],num_updates=10,clip_eps=0.2,lambd=0.97,batch_size=64,buffer_size=2048,coef_ent=0.0,max_grad_norm=0.5):
        super(PPO, self).__init__(device, discount_factor)
        self.env = env

        self.num_state = env.observation_space.shape
        self.num_action = env.action_space.shape

        self.actor = ActorNetwork(self.num_state, self.num_action).to(self.device)
        self.critic = CriticNetwork(self.num_state).to(self.device)

        self.optimzer_actor = optim.Adam(self.actor.parameters(), lr=lr_rate[0])
        self.optimzer_critic = optim.Adam(self.critic.parameters(), lr=lr_rate[1])

        self.rollbuffer = RolloutBuffer(buffer_size=buffer_size,observation_space=env.observation_space,action_space=env.action_space,device=device)

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_updates = num_updates
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def is_update(self,steps):
        return steps % self.buffer_size == 0

    def step(self,state,t,steps):
        t += 1

        action,log_pi,value = self.explore(state)
        next_state,reward,done,_ = self.env.step(action)

        if t == self.env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done


        self.rollbuffer.add(state,action,reward,done_masked,value,log_pi)

        if steps % self.buffer_size == 0:
            last_state = th.tensor(next_state, dtype=th.float, device=self.device).view(-1, *self.num_state)
            with torch.no_grad():
                last_value = self.critic(last_state)
            self.rollbuffer.compute_returns_and_advantage(last_value,done)

        if done:
            t = 0
            next_state = self.env.reset()

        return next_state,t

    def update(self):
        for _ in range(self.num_updates):
            for transitions in self.rollbuffer.get(batch_size=self.batch_size):
                state_batch = th.tensor(transitions.obs, device=self.device, dtype=th.float)
                action_batch = th.tensor(transitions.actions, device=self.device, dtype=th.float)
                return_batch = th.tensor(transitions.returns, device=self.device, dtype=th.float)
                logpi_old_batch = th.tensor(transitions.old_log_prob, device=self.device, dtype=th.float)
                advantages_batch = th.tensor(transitions.advantages, device=self.device, dtype=th.float)

                critic_loss =  (self.critic(state_batch).squeeze_(1) - return_batch).pow_(2).mean()
                self.optimzer_critic.zero_grad()
                critic_loss.backward(retain_graph=False)
                nn.utils.clip_grad_norm(self.critic.parameters(),self.max_grad_norm)
                self.optimzer_critic.step()

                log_pis = self.actor.evaluate_log_pi(state_batch,action_batch).squeeze_(1)
                mean_entropy = -log_pis.mean()
                ratios = (log_pis-logpi_old_batch).exp_()
                loss_actor1 = -ratios*advantages_batch
                loss_actor2 = -th.clamp(ratios,
                                        1-self.clip_eps,
                                        1+self.clip_eps) * advantages_batch
                actor_loss = th.max(loss_actor1,loss_actor2).mean() - self.coef_ent * mean_entropy
                self.optimzer_actor.zero_grad()
                actor_loss.backward(retain_graph=False)
                nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
                self.optimzer_actor.step()
        self.rollbuffer.reset()


    def explore(self,state):
        state = th.tensor(state,dtype=th.float,device=self.device).view(-1,*self.num_state)
        with torch.no_grad():
            action,log_pi = self.actor.sample(state)
            value = self.critic(state)
        return action.cpu().numpy()[0],log_pi.item(),value

    def exploit(self,state):
        state = th.tensor(state, dtype=th.float, device=self.device).view(-1, *self.num_state)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

