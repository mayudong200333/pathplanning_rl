import torch as th
import warnings
import numpy as np
from gym import spaces
from abc import ABC,abstractmethod
from typing import Any,Dict,Generator,List,Optional,Union
from algorithm.common.preprocessing import get_obs_shape,get_action_dim

class BaseBuffer(ABC):
    def __init__(self,
                 buffer_size:int,
                 observation_space:spaces.Space,
                 action_space:spaces.Space,
                 device:Union[th.device,str]="cpu"):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space

        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)

        self.full = False
        self.pos = 0
        self.device = device

    def size(self):
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self,*args,**kwargs):
        raise NotImplementedError()

    def extend(self,*args,**kwargs):
        for data in zip(*args):
            self.add(*data)

    def reset(self):
        self.pos = 0
        self.full = False

    def sample(self,batch_size,env):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0,upper_bound,size=batch_size)
        return

    @abstractmethod
    def _get_samples(self,batch_inds,env):
        return



