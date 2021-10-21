import envs
import gym
import numpy as np
from utils.logger import Logger
from utils.visualize import Visualize
from maps.example import WALLS


if __name__ == '__main__':
    env_name = 'single-basic2duavenv-v0'
    env = gym.make(env_name)
    obs = env.reset()
    goal = env._get_goal()
    state = env._get_position()
    logger = Logger(goal=goal)
    logger.log(state, None)
    r = 0
    for i in range(200):
        act = env.action_space.sample()
        obs, rew, done, _ = env.step(act)
        state = env._get_position()
        print(obs)
        logger.log(state, act)
        r += rew
        if done :
            break

    print('ok')
    env.close()
    print(logger.history)
    vis = Visualize(map_matrix=WALLS['FourRooms'], history=logger.history)








