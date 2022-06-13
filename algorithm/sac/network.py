import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    def __init__(self,obs_shape,action_dim,hidden_size=256):
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
            nn.Linear(num_state,hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,2*num_action),
        )

    def forward(self,states):
        return th.tanh(self.net(states).chunk(2,dim=-1)[0])

    def sample(self,states):
        means,log_stds = self.net(states).chunk(2,dim=-1)
        return self.reparameterize(means,log_stds.clamp_(-20,2))

    def reparameterize(self,means,log_stds):
        stds = log_stds.exp()
        noises = torch.randn_like(means)
        us = means + noises * stds
        actions = th.tanh(us)
        log_pis = self._calculate_log_pi(log_stds,noises,actions)
        return actions,log_pis

    @staticmethod
    def _calculate_log_pi(log_stds,noises,actions):
        stds = log_stds.exp()
        gaussian_log_probs = Normal(th.zeros_like(stds),stds).log_prob(stds*noises).sum(dim=-1,keepdim=True)
        log_pis = gaussian_log_probs - th.log(1-actions.pow(2)+1e-6).sum(dim=-1,keepdim=True)
        return log_pis

class CriticNetwork(nn.Module):
    def __init__(self,obs_shape,action_dim,hidden_size=256):
        super().__init__()
        if isinstance(obs_shape, int):
            num_state = obs_shape
        else:
            num_state = obs_shape[0]

        if isinstance(action_dim, int):
            num_action = action_dim
        else:
            num_action = action_dim[0]

        self.net1 = nn.Sequential(
            nn.Linear(num_state+num_action, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.net2 = nn.Sequential(
            nn.Linear(num_state+num_action, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self,states,actions):
        x = th.cat([states,actions],dim=-1)
        return self.net1(x),self.net2(x)