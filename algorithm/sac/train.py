from algorithm.common.base_class import Trainer
from algorithm.sac.sac import SAC
import gym
import envs
import pybullet_envs

if __name__ == '__main__':
    env_id = 'InvertedPendulumBulletEnv-v0'
    seed = 0
    num_steps = 5 * 10 ** 4
    eval_interval = 10 ** 3

    env = gym.make(env_id)
    test_env = gym.make(env_id)
    algo = SAC(env=env)

    trainer = Trainer(
        env=env,
        env_test=test_env,
        algo=algo,
        seed=seed,
        eval_interval=eval_interval
    )

    trainer.train()