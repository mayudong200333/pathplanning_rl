import envs
import gym
import numpy as np

if __name__ == '__main__':
    env_name = 'single-basicgridenv-v0'
    env = gym.make(env_name)
    obs = env.reset()
    r = 0
    while True:
        act = env.action_space.sample()
        obs, rew, done, _ = env.step(act)
        r += rew
        if done :
            break
    print(r)

    a = np.array([[]])
    a.append(2)
    print(a)



