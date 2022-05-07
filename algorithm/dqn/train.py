from algorithm.common.base_class import Trainer
from algorithm.dqn.dqn import DQN
import gym
import envs

if __name__ == '__main__':
    env_id = 'CartPole-v0'
    seed = 0
    num_steps = 5*10**4
    eval_interval = 10**3

    env = gym.make(env_id)
    test_env = gym.make(env_id)
    algo = DQN(env=env)

    trainer = Trainer(
        env = env,
        env_test= test_env,
        algo = algo,
        seed = seed,
        eval_interval=eval_interval
    )

    trainer.train()