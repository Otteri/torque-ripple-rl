from gym.envs.registration import register

register(id='IlmarinenEnv-v0',
    entry_point='envs.ilmarinen_env_dir:IlmarinenEnv',
    kwargs={"visual": True, "compensation": True}
)

register(id='IlmarinenEnv2-v0',
    entry_point='envs.ilmarinen_env_dir:IlmarinenEnv',
    kwargs={"visual": False, "compensation": True}
)

# Can be used to show how works without compnesation
register(id='IlmarinenEnv3-v0',
    entry_point='envs.ilmarinen_env_dir:IlmarinenEnv',
    kwargs={"visual": True, "compensation": False}
)