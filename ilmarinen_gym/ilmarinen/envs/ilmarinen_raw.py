from gym import Env

# Create direct linkage to C++ <-> Python API
# To access:
#  Ilmarinen = gym.make('IlmarinenRawQlr-v2')
#  sim = Ilmarinen.api
#
# Import Ilmarinen in init to avoid namespace collisions with various versions.
# If this is not done, then the program may interestingly segfaults when it tries to
# use the first Ilmarinen from path. "Import as" cannot be used to solve the problem.
class IlmarinenRawILC(Env):
    def __init__(self):
        from ilmarinen.envs.v1 import Ilmarinen as Ilmarinen_v1
        self.api = Ilmarinen_v1.SandboxApi()

# Version used for torque simulations
class IlmarinenRawQlrV1(Env):
    def __init__(self):
        from ilmarinen.envs.v2 import Ilmarinen as Ilmarinen_v3
        self.api = Ilmarinen_v3.SandboxApi()

# This version has setNoise() and episode length is sligthly different
class IlmarinenRawQlrV2(Env):
    def __init__(self):
        from ilmarinen.envs.v3 import Ilmarinen as Ilmarinen_v3
        self.api = Ilmarinen_v3.SandboxApi()