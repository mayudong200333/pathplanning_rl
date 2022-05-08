import torch as th
import warnings
import numpy as np
import psutil
from gym import spaces
from abc import ABC,abstractmethod
from typing import Any,Dict,Generator,List,Optional,Union
from algorithm.common.preprocessing import get_obs_shape,get_action_dim
from algorithm.common.type_aliases import ReplayBufferSamples,RolloutBufferSamples

class BaseBuffer(ABC):
    def __init__(self,
                 buffer_size:int,
                 observation_space:spaces.Space,
                 action_space:spaces.Space,
                 device:Union[th.device,str]=th.device('cuda:0' if th.cuda.is_available() else 'cpu')):
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
        return self._get_samples(batch_inds,env=env)

    @abstractmethod
    def _get_samples(self,batch_inds,env):
        raise NotImplementedError()

    def to_torch(self,array:np.ndarray,copy:bool=True):
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

class ReplayBuffer(BaseBuffer):

    def __init__(self,buffer_size:int,
                 observation_space:spaces.Space,
                 action_space:spaces.Space,
                 device:Union[th.device,str]=th.device('cuda:0' if th.cuda.is_available() else 'cpu'),
                 optimize_memory_usage:bool = False):
        super(ReplayBuffer,self).__init__(buffer_size,observation_space,action_space,device)

        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros((self.buffer_size,)+self.obs_shape,dtype=observation_space.dtype)

        if optimize_memory_usage:
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size,)+self.obs_shape,dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size,self.action_dim),dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size,1),dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,1),dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(self,
            obs:np.ndarray,
            next_obs:np.ndarray,
            action:np.ndarray,
            reward:np.ndarray,
            done:np.ndarray):

        self.observations[self.pos] = np.array(obs).copy()
        if self.optimize_memory_usage:
            self.observations[(self.pos+1)%self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self,batch_size:int,env):
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size,env=env)

        if self.full:
            batch_inds = (np.random.randint(1,self.buffer_size,size=batch_size)+self.pos)%self.buffer_size
        else:
            batch_inds = np.random.randint(0,self.pos,size=batch_size)

        return self._get_samples(batch_inds,env=env)

    def _get_samples(self,batch_inds:np.ndarray,env):
        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds+1)%self.buffer_size,:]
        else:
            next_obs = self.next_observations[batch_inds,:]

        data = (
            self.observations[batch_inds,:],
            self.actions[batch_inds,:],
            next_obs,
            self.dones[batch_inds,:],
            self.rewards[batch_inds,:]
        )

        return ReplayBufferSamples(*tuple(map(self.to_torch,data)))


class RolloutBuffer(BaseBuffer):

    def __init__(self,buffer_size:int,
                 observation_space:spaces.Space,
                 action_space:spaces.Space,
                 device:Union[th.device,str]=th.device('cuda:0' if th.cuda.is_available() else 'cpu',),
                 gae_lambda:float = 1,
                 gamma:float = 1):
        super().__init__(buffer_size,observation_space,action_space,device)
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.observations,self.actions,self.rewards,self.advantages = None,None,None,None
        self.returns,self.values,self.log_probs,self.dones = None,None,None,None
        self.reset()

    def reset(self):

        self.observations = np.zeros((self.buffer_size,)+self.obs_shape,dtype=np.float32)
        self.actions = np.zeros((self.buffer_size,self.action_dim),dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,1),dtype=np.float32)
        self.values = np.zeros((self.buffer_size,1),dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,1),dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,1),dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        super().reset()

    def compute_returns_and_advantage(self,last_values:th.Tensor,done:bool):

        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - done
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step+1]
                next_values = self.values[step+1]

            delta = self.rewards[step] + self.gamma * next_values * (1-self.dones[step]) - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(self,
            obs:np.ndarray,
            action:np.ndarray,
            reward:np.ndarray,
            done:np.ndarray,
            value:th.Tensor,
            log_prob:th.Tensor):

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self,batch_size:Optional[int]=None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)

        if batch_size is None:
            batch_size = self.buffer_size
        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx:start_idx+batch_size])
            start_idx += batch_size


    def _get_samples(self,batch_inds:np.ndarray,env=None):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.dones[batch_inds],
            self.returns[batch_inds].flatten(),
            self.rewards[batch_inds]
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch,data)))






