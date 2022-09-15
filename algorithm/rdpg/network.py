import numpy as np
import math
import torch as th
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

FLOAT = th.cuda.FloatTensor

class ActorNetwork(nn.Module):
    def __init__(self,obs_shape,action_dim,hidden_size=(400,300,300),init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.hidden_size = hidden_size
        if isinstance(obs_shape, int):
            num_state = obs_shape
        else:
            num_state = obs_shape[0]

        if isinstance(action_dim, int):
            num_action = action_dim
        else:
            num_action = action_dim[0]

        self.fc1 = nn.Linear(num_state,hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0],hidden_size[1])
        self.lstm = nn.LSTMCell(hidden_size[1],hidden_size[2])
        self.fc3 = nn.Linear(hidden_size[2],num_action)

        nn.init.uniform_(self.fc1.weight, -1. / np.sqrt(self.fc1.weight.data.size()[0]),
                         1. / np.sqrt(self.fc1.weight.data.size()[0]))
        nn.init.uniform_(self.fc2.weight, -1. / np.sqrt(self.fc2.weight.data.size()[0]),
                         1. / np.sqrt(self.fc2.weight.data.size()[0]))
        nn.init.uniform_(self.fc3.weight, -init_w, init_w)

        self.cx = Variable(th.zeros(1,hidden_size[2])).type(FLOAT)
        self.hx = Variable(th.zeros(1,hidden_size[2])).type(FLOAT)

    def reset_lstm_hidden_states(self):
        self.cx = Variable(th.zeros(1, self.hidden_size[2])).type(FLOAT)
        self.hx = Variable(th.zeros(1, self.hidden_size[2])).type(FLOAT)

    def forward(self,x,hidden_states=None):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        if hidden_states == None:
            hx,cx = self.lstm(h,(self.hx,self.cx))
            self.hx = hx
            self.cx = cx
        else:
            hx,cx = self.lstm(h,hidden_states)


        h = hx.clone().detach()
        h = self.fc3(h)
        y = F.tanh(h)

        return y,(hx,cx)


class CriticNetwork(nn.Module):

    def __init__(self,obs_shape,action_dim,hidden_size=(400,300,300),init_w=3e-3):
        super(CriticNetwork,self).__init__()
        self.hidden_size = hidden_size
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
        self.lstm = nn.LSTMCell(hidden_size[1], hidden_size[2])
        self.fc3 = nn.Linear(hidden_size[2], 1)

        nn.init.uniform_(self.fc1.weight, -1. / np.sqrt(self.fc1.weight.data.size()[0]),
                         1. / np.sqrt(self.fc1.weight.data.size()[0]))
        nn.init.uniform_(self.fc2.weight, -1. / np.sqrt(self.fc2.weight.data.size()[0]),
                         1. / np.sqrt(self.fc2.weight.data.size()[0]))
        nn.init.uniform_(self.fc3.weight, -init_w, init_w)

        self.cx = Variable(th.zeros(1, hidden_size[2])).type(FLOAT)
        self.hx = Variable(th.zeros(1, hidden_size[2])).type(FLOAT)

    def reset_lstm_hidden_states(self):
        self.cx = Variable(th.zeros(1, self.hidden_size[2])).type(FLOAT)
        self.hx = Variable(th.zeros(1, self.hidden_size[2])).type(FLOAT)

    def forward(self,x,action,hidden_states=None):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(th.cat([h,action],dim=1)))
        if hidden_states == None:
            hx,cx = self.lstm(h,(self.hx,self.cx))
            self.hx = hx
            self.cx = cx
        else:
            hx,cx = self.lstm(h,hidden_states)

        h = hx.clone().detach()
        y = self.fc3(h)
        return y,(hx,cx)







