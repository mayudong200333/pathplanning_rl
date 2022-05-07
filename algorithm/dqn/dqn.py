import copy

from algorithm.common.base_class import Algorithm
from algorithm.common.noise import OrnsteinUhlenbeckProcess
from algorithm.dqn.network import QNetwork
from algorithm.common.buffer import ReplayBuffer

import torch as th
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class DQN(Algorithm):
    def __init__(self,env,device=th.device('cuda:0' if th.cuda.is_available() else 'cpu'),discount_factor=0.99,
                 lr_rate=0.001,batch_size=32,buffer_size=int(5e4)):
        super(DQN, self).__init__(device, discount_factor)
        self.env = env

        self.num_state = env.observation_space.shape
        self.num_action = env.action_space.n

        self.q = QNetwork(self.num_state,self.num_action).to(device)
        self.q_target = copy.deepcopy(self.q)
        self.optimizer = optim.Adam(self.q.parameters(),lr=lr_rate)

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, observation_space=env.observation_space,
                                          action_space=env.action_space,
                                          device=device)

        self.batch_size = batch_size

        state = env.reset()
        while not self.replay_buffer.full:
            action = self.env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            self.replay_buffer.add(state, next_state, action, reward, done)
            state = env.reset() if done else next_state
        print('{} Data collected'.format(buffer_size))

    def update(self):
        transitions = self.replay_buffer.sample(batch_size=self.batch_size, env=self.env)

        state_batch = th.tensor(transitions.obs, device=self.device, dtype=th.float)
        action_batch = th.tensor(transitions.actions, device=self.device, dtype=th.int)

        next_state_batch = th.tensor(transitions.next_obs, device=self.device, dtype=th.float)
        reward_batch = th.tensor(transitions.rewards, device=self.device, dtype=th.float)
        done_batch = th.tensor(transitions.dones, device=self.device, dtype=th.float)

        q = self.q(state_batch)
        q_target = copy.deepcopy(q.data)
        maxq = th.max(th.tensor(self.q_target(next_state_batch),dtype=th.float),dim=1,).values
        for i in range(self.batch_size):
            q_target[i,action_batch[i].data.cpu().numpy()[0]] = reward_batch[i] + self.discount_factor*maxq[i]*(1.-done_batch[i])
        self.optimizer.zero_grad()

        loss = F.mse_loss(q,q_target)
        loss.backward()
        self.optimizer.step()

        self.q_target = copy.deepcopy(self.q)

    def is_update(self,steps):
        return True

    def exploit(self,state):
        state_tensor = th.tensor(state, dtype=th.float, device=self.device).view(-1, *self.num_state)
        action = th.argmax(self.q(state_tensor)).item()
        return action

    def explore(self,state,steps):
        epsilon = 0.7 * (1/(steps+1))
        if epsilon <= np.random.uniform(0,1):
            action = self.exploit(state)
        else:
            action = np.random.choice(self.num_action)
        return action

    def step(self,state,t,steps):
        t += 1

        action = self.explore(state,steps)
        next_state,reward,done,_ = self.env.step(action)
        self.replay_buffer.add(state, next_state, action, reward, done)
        if done:
            t = 0
            next_state = self.env.reset()

        return next_state,t








