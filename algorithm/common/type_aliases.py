from typing import NamedTuple
import gym
import numpy as np
import torch as th

class ReplayBufferSamples(NamedTuple):
    obs:th.Tensor
    actions:th.Tensor
    next_obs:th.Tensor
    dones:th.Tensor
    rewards:th.Tensor