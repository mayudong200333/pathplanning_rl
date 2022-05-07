import numpy as np
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):

    def __init__(self,obs_shape,action_dim,hidden_size=(400,300),init_w=3e-3):
        super(ActorNetwork,self).__init__()
        if isinstance(obs_shape,int):
            num_state = obs_shape
        else:
            num_state = obs_shape[0]

        if isinstance(action_dim,int):
            num_action = action_dim
        else:
            num_action = action_dim[0]

        self.fc1 = nn.Linear(num_state,hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0],hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1],num_action)

        nn.init.uniform_(self.fc1.weight, -1. / math.sqrt(self.fc1.weight.data.size()[0]),
                         1. / math.sqrt(self.fc1.weight.data.size()[0]))
        nn.init.uniform_(self.fc2.weight, -1. / math.sqrt(self.fc2.weight.data.size()[0]),
                         1. / math.sqrt(self.fc2.weight.data.size()[0]))
        nn.init.uniform_(self.fc3.weight,-init_w,init_w)

    def forward(self,x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = th.tanh(self.fc3(h))
        return y

class CriticNetwork(nn.Module):

    def __init__(self,obs_shape,action_dim,hidden_size=(400,300),init_w=3e-3):
        super(CriticNetwork,self).__init__()
        if isinstance(obs_shape,int):
            num_state = obs_shape
        else:
            num_state = obs_shape[0]

        if isinstance(action_dim,int):
            num_action = action_dim
        else:
            num_action = action_dim[0]

        self.fc1 = nn.Linear(num_state, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0]+num_action, hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], 1)

        nn.init.uniform_(self.fc1.weight, -1. / np.sqrt(self.fc1.weight.data.size()[0]),
                         1. / np.sqrt(self.fc1.weight.data.size()[0]))
        nn.init.uniform_(self.fc2.weight, -1. / np.sqrt(self.fc2.weight.data.size()[0]),
                         1. / np.sqrt(self.fc2.weight.data.size()[0]))
        nn.init.uniform_(self.fc3.weight, -init_w, init_w)

    def forward(self,x,action):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(th.cat([h,action],dim=1)))
        y = self.fc3(h)
        return y




