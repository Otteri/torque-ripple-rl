from gym.envs.registration import register

# Allow direct access to C++ <-> Python interface, as it would be
# tiring to duplicate the whole interface to gym.
register(
    id='IlmarinenRawILC-v0',
    entry_point='envs.ilmarinen_env_dir:IlmarinenRawILC'
)

register(
    id='IlmarinenRawQlr-v1',
    entry_point='envs.ilmarinen_env_dir:IlmarinenRawQlrV1'
)

register(
    id='IlmarinenRawQlr-v2',
    entry_point='envs.ilmarinen_env_dir:IlmarinenRawQlrV2'
)

# Typical Gym interface. Built on top of the previous interface
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
