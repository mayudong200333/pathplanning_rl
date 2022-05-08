import numpy as np
import math

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    def __init__(self,obs_shape,action_dim):
        super().__init__()
        if isinstance(obs_shape,int):
            num_state = obs_shape
        else:
            num_state = obs_shape[0]

        if isinstance(action_dim,int):
            num_action = action_dim
        else:
            num_action = action_dim[0]

        self.net = nn.Sequential(
            nn.Linear(num_state,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,num_action),
        )

        self.log_stds = nn.Parameter(th.zeros(1,num_action))

    def forward(self,states):
        return th.tanh(self.net(states))

    def sample(self,states):
        return self.reparameterize(self.net(states),self.log_stds)

    def evaluate_log_pi(self,states,actions):
        noises = (self.atanh(actions)-self.net(states))/(self.log_stds.exp()+1e-8)
        return self._calculate_log_pi(self.log_stds,noises,actions)

    def reparameterize(self,means,log_stds):
        stds = log_stds.exp()
        noises = torch.randn_like(means)
        us = means + noises * stds
        actions = th.tanh(us)
        log_pis = self._calculate_log_pi(log_stds,noises,actions)
        return actions,log_pis

    @staticmethod
    def atanh(x):
        return 0.5*(th.log(1+x+1e-6)-th.log(1-x+1e-6))

    @staticmethod
    def _calculate_log_pi(log_stds,noises,actions):
        stds = log_stds.exp()
        gaussian_log_probs = Normal(th.zeros_like(stds),stds).log_prob(stds*noises).sum(dim=-1,keepdim=True)
        log_pis = gaussian_log_probs - th.log(1-actions.pow(2)+1e-6).sum(dim=-1,keepdim=True)
        return log_pis


class CriticNetwork(nn.Module):
    def __init__(self,obs_shape):
        super().__init__()
        if isinstance(obs_shape, int):
            num_state = obs_shape
        else:
            num_state = obs_shape[0]

        self.net = nn.Sequential(
            nn.Linear(num_state, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self,states):
        return self.net(states)
