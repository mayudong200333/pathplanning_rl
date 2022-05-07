## train the env
import gym
import torch.nn

import envs
import argparse
import numpy as np
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback,StopTrainingOnRewardThreshold

from maps.example import WALLS


if __name__ == '__main__':


    parse = argparse.ArgumentParser(description='Single agent reinforcement learning in UAV path planning')
    parse.add_argument('--env',default='single-basic2duavenv-v1',type=str)
    parse.add_argument('--algo',default='ddpg',type=str)
    parse.add_argument('--num', default='10', type=str)
    parse.add_argument('--map', default='random', type=str)
    ARGS = parse.parse_args()

    #### Save directory #####
    filename = 'results/' + ARGS.env + '-' + ARGS.algo + '-' + ARGS.map + '-' + ARGS.num

    #### Train the env #####
    env_name = ARGS.env
    sa_env_kwargs = dict(threshold_distance=0.1)
    train_env = make_vec_env(env_name, env_kwargs=sa_env_kwargs, n_envs=1)

    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space", train_env.observation_space)

    onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512, dict(vf=[256, 128], pi=[256, 128])])
    offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512, 256, 128])

    n_actions = train_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    #model = PPO(a2cppoMlpPolicy, train_env,policy_kwargs=onpolicy_kwargs, learning_rate=3e-5,target_kl=15e-4,tensorboard_log=filename + '/tb/', verbose=1)
    model = DDPG(td3ddpgMlpPolicy, train_env, action_noise=action_noise,policy_kwargs=offpolicy_kwargs, tensorboard_log=filename + '/tb/', verbose=1)

    eval_env = gym.make(env_name,threshold_distance=0.1)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=10000, verbose=1)

    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1,
                                 best_model_save_path=filename + '/',
                                 log_path=filename + '/',
                                 eval_freq=2000,
                                 deterministic=True,
                                 render=False)

    model.learn(total_timesteps=1e6,
                callback=eval_callback,
                log_interval=100)

    ### Save the model ###
    model.save(filename + '/success_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename + '/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j]) + "," + str(data['results'][j][0][0]))






