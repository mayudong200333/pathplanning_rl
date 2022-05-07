import argparse
import re
import numpy as np
import gym
import envs
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from maps.example import WALLS
from utils.logger import Logger
from utils.visualize import Visualize

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as pat

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Single agent test in reinforcement learning of path planning')
    parse.add_argument('--env', default='single-basic2duavenv-v1', type=str)
    parse.add_argument('--algo', default='ddpg', type=str)
    parse.add_argument('--num', default='10', type=str)
    parse.add_argument('--map', default='random', type=str)
    ARGS = parse.parse_args()

    filename = 'results/' + ARGS.env + '-' + ARGS.algo + '-' + ARGS.map + '-' + ARGS.num

    path = filename + '/success_model.zip'

    #model = PPO.load(path)
    model = DDPG.load(path)

    env_name = ARGS.env

    #### Evaluate the model ####################################
    '''

    eval_env = gym.make(env_name,
                        )
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    
    '''


    #### Show, record a video, and log the model's performance #
    test_env = gym.make(env_name,
                        )

    obs = test_env.reset()
    state = test_env._get_position()
    goal = test_env._get_goal()
    logger = Logger(goal=goal)
    logger.log(state,None)
    r = 0
    while True:  # Up to 6''
        action, _states = model.predict(obs,
                                        deterministic=False  # OPTIONAL 'deterministic=False'
                                        )


        obs, reward, done, info = test_env.step(action)
        #print(action)
        state = test_env._get_position()
        logger.log(state,action)
        r += reward
        if done :
            break
    test_env.close()
    print(r)
    vis = Visualize(map_matrix=test_env._map,history=logger.history)
