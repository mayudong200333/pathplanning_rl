from gym.envs.registration import register

register(
    id='single-basicgridenv-v0',
    entry_point='envs.BasicGridEnv:BasicGridEnv',
)

register(
    id='single-basic2duavenv-v0',
    entry_point='envs.Basic2dUAVEnv:Basic2dUAVEnv',
)

register(
    id='single-basic2duavenv-v1',
    entry_point='envs.Basic2dUAVEnv2:Basic2dUAVEnv2',
)