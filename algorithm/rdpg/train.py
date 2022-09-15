from algorithm.common.base_class import Trainer
from algorithm.rdpg.rdpg import RDPG
import gym
import envs

if __name__ == '__main__':
    env_id = 'Pendulum-v0'
    seed = 0
    num_steps = 1*10**7
    eval_interval = 10**3

    env = gym.make(env_id)
    test_env = gym.make(env_id)
    algo = RDPG(env=env)

    trainer = Trainer(
        env = env,
        env_test= test_env,
        algo = algo,
        seed = seed,
        eval_interval=eval_interval
    )

    trainer.train()