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

class RolloutBufferSamples(NamedTuple):
    obs:th.Tensor
    actions:th.Tensor
    old_values:th.Tensor
    old_log_prob:th.Tensor
    advantages:th.Tensor
    dones:th.Tensor
    returns:th.Tensor
    rewards:th.Tensor