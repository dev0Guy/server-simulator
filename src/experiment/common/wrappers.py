from gymnasium.vector.utils import spaces
import gymnasium as gym
import numpy as np

class FlattenActionWrapper(gym.ActionWrapper):
    """Converts Tuple(Discrete(2), Tuple(Discrete(2), Discrete(10)))
       → MultiDiscrete([2, 2, 10]) so SB3 can handle it."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete([2, 2, 10])

    def action(self, action: np.ndarray):
        # action is [a0, a1, a2] — reconstruct the original nested tuple
        return int(action[0]), (int(action[1]), int(action[2]))

    def render(self):
        return self.env.render()
