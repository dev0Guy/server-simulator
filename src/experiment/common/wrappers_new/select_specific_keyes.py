import gymnasium.spaces
from gymnasium import ObservationWrapper, spaces
from gymnasium.core import ObsType, WrapperObsType


class SelectSpecificKeyesWrapper(ObservationWrapper):

    def __init__(self, env, *keys: str):
        """Constructor for the observation wrapper."""
        super().__init__(env)
        self._keys = set(keys)

        observation_space = gymnasium.spaces.Dict({
            k: v
            for k, v in self.env.observation_space.items()
            if k in self._keys
        })
        self.observation_space = observation_space


    def observation(self, observation: ObsType) -> WrapperObsType:
        return {
            k: v
            for k, v in observation.items()
            if k in self._keys
        }
