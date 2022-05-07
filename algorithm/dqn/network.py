import numpy as np
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    # The action for DQN must be Discrete
    def __init__(self,obs_shape,action_num,hidden_size=16):
        super(QNetwork,self).__init__()
        if isinstance(obs_shape,int):
            num_state = obs_shape
        else:
            num_state = obs_shape[0]

        self.fc1 = nn.Linear(num_state,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.fc4 = nn.Linear(hidden_size,action_num)

    def forward(self,x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        h = F.elu(self.fc3(h))
        y = F.elu(self.fc4(h))

        return y
