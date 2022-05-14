from gym.envs.registration import register

register(
    id='egraph',
    entry_point='rejoice.envs.egraph_env:EGraphEnv',
    max_episode_steps=300,
)