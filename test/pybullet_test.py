import pybullet as p
import time
import os
import gym
import torch as th
import numpy as np

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.reset()
    print(env.spec.max_episode_steps)
    print(env.action_space.sample())
    num_state = env.observation_space.shape
    num_action = env.action_space.n
    print(num_state)
    print(num_action)
    a = th.tensor([[4,5,6],[1,2,3]])
    b = th.argmax(a)
    print(b)


