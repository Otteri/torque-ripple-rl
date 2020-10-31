from gym.envs.registration import register

# Allow direct access to Ilmarinen for scripts that can
# utilize the interface to full extent, as it would be tiring to
# to duplicate the interface to gym side. The architecture looks like:
# C++ core | python interface | gym interface | python-script
# Gym interface needs only to contain RL related functions.
# Note: Ilmarinen could be imported directly, but since it is now here,
# use: from envs import Ilmarinen, if direct access to python interface is required.
#from envs.ilmarinen_env_dir import Ilmarinen
register(
    id='IlmarinenRawILC-v0',
    entry_point='envs.ilmarinen_env_dir:IlmarinenRawILC'
)

register(
    id='IlmarinenRawQlr-v0',
    entry_point='envs.ilmarinen_env_dir:IlmarinenRawQlr'
)

register(
    id='IlmarinenEnv-v0',
    entry_point='envs.ilmarinen_env_dir:IlmarinenEnv',
    kwargs={"visual": True, "compensation": True}
)

register(
    id='IlmarinenEnv2-v0',
    entry_point='envs.ilmarinen_env_dir:IlmarinenEnv',
    kwargs={"visual": False, "compensation": True}
)

# Can be used to show how works without compnesation
register(
    id='IlmarinenEnv3-v0',
    entry_point='envs.ilmarinen_env_dir:IlmarinenEnv',
    kwargs={"visual": True, "compensation": False}
)
