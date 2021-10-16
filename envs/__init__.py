from gym.envs.registration import register

register(
    id='single-basicgridenv-v0',
    entry_point='envs.BasicGridEnv:BasicGridEnv',
)