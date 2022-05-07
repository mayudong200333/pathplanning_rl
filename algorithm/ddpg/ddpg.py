import copy

from algorithm.common.base_class import Algorithm
from algorithm.common.noise import OrnsteinUhlenbeckProcess
from algorithm.ddpg.network import ActorNetwork,CriticNetwork
from algorithm.common.buffer import ReplayBuffer

import torch as th
import torch.optim as optim
import torch.nn.functional as F
import numpy as np




class DDPG(Algorithm):
    def __init__(self,env,device=th.device('cuda:0' if th.cuda.is_available() else 'cpu'),discount_factor=0.99,
                 lr_rate=[1e-4,1e-3],tau=1e-3,weigt_decay=1e-2,batch_size=64,buffer_size=int(1e6)):
        super(DDPG,self).__init__(device,discount_factor)
        self.env = env

        self.num_state = env.observation_space.shape
        self.num_action = env.action_space.shape

        self.actor = ActorNetwork(self.num_state,self.num_action).to(self.device)
        self.critic = CriticNetwork(self.num_state,self.num_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.optimzer_actor = optim.Adam(self.actor.parameters(),lr=lr_rate[0])
        self.optimzer_critic = optim.Adam(self.critic.parameters(),lr=lr_rate[1],weight_decay=weigt_decay)

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,observation_space=env.observation_space,action_space=env.action_space,
                                          device=device)
        self.batch_size = batch_size
        self.tau = tau

        self.noise = OrnsteinUhlenbeckProcess(size=self.num_action[0])

        state = env.reset()
        while not self.replay_buffer.full:
            action = self.env.action_space.sample()
            next_state,reward,done,_ = env.step(action)
            self.replay_buffer.add(state,next_state,action,reward,done)
            state = env.reset() if done else next_state
        print('{} Data collected'.format(buffer_size))


    def update(self):
        transitions = self.replay_buffer.sample(batch_size=self.batch_size,env=self.env)

        state_batch = th.tensor(transitions.obs,device=self.device,dtype=th.float)
        action_batch = th.tensor(transitions.actions,device=self.device,dtype=th.float)

        next_state_batch = th.tensor(transitions.next_obs,device=self.device,dtype=th.float)
        reward_batch = th.tensor(transitions.rewards,device=self.device,dtype=th.float)
        done_batch = th.tensor(transitions.dones, device=self.device, dtype=th.float)

        q = self.critic(state_batch,action_batch)
        next_q = self.critic_target(next_state_batch,self.actor_target(next_state_batch))
        target_q = reward_batch + self.discount_factor*next_q*(1.-done_batch.data)

        critic_loss = F.mse_loss(q,target_q)
        self.optimzer_critic.zero_grad()
        critic_loss.backward()
        self.optimzer_critic.step()

        actor_loss = -self.critic(state_batch,self.actor(state_batch)).mean()
        self.optimzer_actor.zero_grad()
        actor_loss.backward()
        self.optimzer_actor.step()

        for target_param,param in zip(self.actor_target.parameters(),self.actor.parameters()):
            target_param.data.copy_(target_param*(1.-self.tau)+param.data*self.tau)
        for target_param,param in zip(self.critic_target.parameters(),self.critic.parameters()):
            target_param.data.copy_(target_param*(1.-self.tau)+param.data*self.tau)

    def is_update(self,steps):
        return True

    def explore(self,state):
        state_tensor = th.tensor(state,dtype=th.float,device=self.device).view(-1,*self.num_state)
        action = self.actor(state_tensor)
        return action.squeeze(0).detach().cpu().numpy()

    def exploit(self,state):
        state_tensor = th.tensor(state, dtype=th.float, device=self.device).view(-1, *self.num_state)
        action = self.actor(state_tensor)
        action += th.tensor(self.noise.sample(),dtype=th.float,device=self.device)
        return action.squeeze(0).detach().cpu().numpy()

    def step(self,state,t,steps):
        t += 1

        action = self.explore(state)
        next_state,reward,done,_ = self.env.step(action)
        self.replay_buffer.add(state, next_state, action, reward, done)
        if done:
            t = 0
            next_state = self.env.reset()

        return next_state,t









