from time import time
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC,abstractmethod
import torch



class Trainer:
    def __init__(self,env,env_test,algo,seed=0,num_steps=10**6,eval_interval=10**4,num_eval_episode=3):

        self.env = env
        self.env_test = env_test
        self.algo = algo

        self.env.seed(seed)
        self.env_test.seed(2**31-seed)

        self.his = {'step':[],'reward':[]}

        self.num_steps = num_steps

        self.eval_interval = eval_interval

        self.num_eval_episode = num_eval_episode

    def train(self):

        self.start_time = time()

        t = 0

        state = self.env.reset()

        for steps in range(1,self.num_steps+1):
            state,t = self.algo.step(state,t,steps)

            if self.algo.is_update(steps):
                self.algo.update()

            if steps % self.eval_interval == 0:
                self.evaluate(steps)

    def evaluate(self,steps):

        rewards = []
        for _ in range(self.num_eval_episode):
            state = self.env_test.reset()
            done = False
            episode_reward = 0.0

            while (not done):
                action = self.algo.exploit(state)
                state,reward,done,_ = self.env_test.step(action)
                episode_reward += reward

            rewards.append(episode_reward)

        mean_reward = np.mean(episode_reward)
        self.his['step'].append(steps)
        self.his['reward'].append(mean_reward)

        print(f'Num steps: {steps:<10}  '
              f'Reward:{mean_reward:<5.1f}   '
              f'Time:{self.time}')

    def plot(self):
        fig = plt.figure(figsize=(8,6))
        plt.plot(self.his['step'],self.his['reward'])
        plt.xlabel('Steps',fontsize=24)
        plt.ylabel('Reward',fontsize=24)
        plt.tick_params(labelsize=18)
        plt.title(f'{self.env.unwrapped.spec.id}',fontsize=24)
        plt.tight_layout()

    @property
    def time(self):
        return str(timedelta(seconds=int(time()-self.start_time)))


class Algorithm(ABC):
    def __init__(self,device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),discount_factor=0.99):
        self.device = device
        self.discount_factor = discount_factor

    def update(self):
        raise NotImplementedError()

    def is_update(self,steps):
        raise NotImplementedError()

    def step(self,step,t,steps):
        raise NotImplementedError()

    def explore(self,state):
        raise NotImplementedError()

    def exploit(self,state):
        raise NotImplementedError()


