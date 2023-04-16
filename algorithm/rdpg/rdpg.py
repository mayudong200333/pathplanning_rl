import copy

from algorithm.common.base_class import Algorithm
from algorithm.common.noise import OrnsteinUhlenbeckProcess
from algorithm.rdpg.network import ActorNetwork,CriticNetwork
from algorithm.common.buffer import EpisodicReplayBuffer

import torch as th
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

FLOAT = th.cuda.FloatTensor

class RDPG(Algorithm):
    def __init__(self,env,device=th.device('cuda:0' if th.cuda.is_available() else 'cpu'),discount_factor=0.99,
                 lr_rate=[1e-4,1e-3],tau=1e-3,weight_decay=1e-2,batch_size=64,buffer_size=int(1e4)):
        super(RDPG,self).__init__(device,discount_factor)
        self.env = env
        self.maxlen=50

        self.num_state = env.observation_space.shape
        self.num_action = env.action_space.shape

        self.actor = ActorNetwork(self.num_state, self.num_action).to(self.device)
        self.critic = CriticNetwork(self.num_state, self.num_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.optimzer_actor = optim.Adam(self.actor.parameters(), lr=lr_rate[0])
        self.optimzer_critic = optim.Adam(self.critic.parameters(), lr=lr_rate[1], weight_decay=weight_decay)

        self.replay_buffer = EpisodicReplayBuffer(buffer_size=buffer_size,
                                                  max_episode_size=self.maxlen,
                                                  observation_space=env.observation_space,
                                                  action_space=env.action_space,
                                                  device=device)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.tau = tau

        self.noise = OrnsteinUhlenbeckProcess(size=self.num_action[0])

    def is_update(self,steps):
        return steps >= self.buffer_size

    def step(self,state,t,steps):
        t += 1

        if steps <= self.buffer_size:
            action = self.env.action_space.sample()
        else:
            if (steps - self.buffer_size) %  self.maxlen == 0:
                self.actor.reset_lstm_hidden_states()
            action = self.explore(state)

        next_state,reward,done,_ = self.env.step(action)

        self.replay_buffer.add(state, next_state, action, reward, done)

        if done:
            t = 0
            next_state = self.env.reset()
            self.actor.reset_lstm_hidden_states()

        return next_state,t

    def update(self):
        transitions = self.replay_buffer.sample(batch_size=self.batch_size, env=self.env)
        state_batch,action_batch,next_state_batch,reward_batch,done_batch = transitions.obs,transitions.actions,transitions.next_obs,transitions.rewards,transitions.dones

        total_actor_loss = 0
        total_critic_loss = 0

        cx_critic = Variable(th.zeros(self.batch_size, self.critic.hidden_size[2])).type(FLOAT)
        hx_critic = Variable(th.zeros(self.batch_size, self.critic.hidden_size[2])).type(FLOAT)

        cx_critic_t = Variable(th.zeros(self.batch_size, self.critic.hidden_size[2])).type(FLOAT)
        hx_critic_t = Variable(th.zeros(self.batch_size, self.critic.hidden_size[2])).type(FLOAT)

        cx_actor = Variable(th.zeros(self.batch_size, self.actor.hidden_size[2])).type(FLOAT)
        hx_actor = Variable(th.zeros(self.batch_size, self.actor.hidden_size[2])).type(FLOAT)

        cx_actor_t = Variable(th.zeros(self.batch_size, self.actor.hidden_size[2])).type(FLOAT)
        hx_actor_t = Variable(th.zeros(self.batch_size, self.actor.hidden_size[2])).type(FLOAT)

        for t in range(self.maxlen):

            state = state_batch[:,t,:]
            action = action_batch[:,t,:]
            next_state = next_state_batch[:,t,:]
            done = done_batch[:,t,:]
            reward = reward_batch[:,t,:]
            target_mu_t,(hx_actor_t,cx_actor_t)= self.actor_target(next_state,(hx_actor_t,cx_actor_t))
            target_q,(_,_) = self.critic_target(next_state,target_mu_t,(hx_actor_t,cx_actor_t))
            yit = reward + self.discount_factor * target_q * (1. - done)
            qit,(_,_) = self.critic(state,action,(hx_actor,cx_actor))
            total_critic_loss +=  F.mse_loss(qit,yit)

            mu_t,(_,_) = self.actor(state,(hx_actor,cx_actor))
            actor_loss_it,(hx_actor,cx_actor) = self.critic(state,mu_t,(hx_actor,cx_actor))
            total_actor_loss -= actor_loss_it.mean()


        total_critic_loss /=  self.maxlen
        self.optimzer_critic.zero_grad()
        total_critic_loss.backward()
        self.optimzer_critic.step()

        total_actor_loss /= self.maxlen
        self.optimzer_actor.zero_grad()
        total_actor_loss.backward()
        self.optimzer_actor.step()

        for target_param,param in zip(self.actor_target.parameters(),self.actor.parameters()):
            target_param.data.copy_(target_param*(1.-self.tau)+param.data*self.tau)
        for target_param,param in zip(self.critic_target.parameters(),self.critic.parameters()):
            target_param.data.copy_(target_param*(1.-self.tau)+param.data*self.tau)


    def explore(self,state):
        state_tensor = th.tensor(state,dtype=th.float,device=self.device).view(-1,*self.num_state)
        action,(_,_) = self.actor(state_tensor)
        return action.squeeze(0).detach().cpu().numpy()

    def exploit(self,state):
        state_tensor = th.tensor(state, dtype=th.float, device=self.device).view(-1, *self.num_state)
        action,(_,_)= self.actor(state_tensor)
        action += th.tensor(self.noise.sample(),dtype=th.float,device=self.device)
        return action.squeeze(0).detach().cpu().numpy()