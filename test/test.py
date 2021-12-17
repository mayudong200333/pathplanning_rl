import envs
import gym
import numpy as np
from utils.logger import Logger
from utils.visualize import Visualize
from maps.example import WALLS


if __name__ == '__main__':
    env_name = 'single-basic2duavenv-v1'
    env = gym.make(env_name)
    obs = env.reset()
    goal = env._get_goal()
    state = env._get_position()
    logger = Logger(goal=goal)
    logger.log(state, None)
    r = 0.0
    for i in range(1000):
        act = env.action_space.sample()
        #act = np.array([1,1])
        obs, rew, done, _ = env.step(act)
        state = env._get_position()
        logger.log(state, act)
        r += rew
        if done :
            break
    print(r)
    print('ok')
    env.close()
    vis = Visualize(map_matrix=WALLS['Random'], history=logger.history)








