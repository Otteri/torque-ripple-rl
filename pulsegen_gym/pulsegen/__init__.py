from gym.envs.registration import register

register(
    id='FourierSeries-v0',
    entry_point='pulsegen.envs:FourierSeries',
)
