from gymnasium.vector.utils import spaces
import gymnasium as gym
import numpy as np

from src.server_simulator.envs import BasicClusterEnv
from src.server_simulator.envs.cluster_simulator.actions import DilationEnvironmentAction, EnvironmentAction


# TODO: Make These wrapper more general
class FlattenActionWrapper(gym.ActionWrapper):
    """Converts Tuple(Discrete(2), Tuple(Discrete(2), Discrete(10)))
       → MultiDiscrete([2, 2, 10]) so SB3 can handle it."""

    def __init__(self, env: BasicClusterEnv):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete([env.action_space[0].n , env.action_space[1][0].n, env.action_space[1][1].n])

    def action(self, action: np.ndarray):
        return EnvironmentAction(
            should_schedule=action[0],
            schedule=(
                action[1],
                action[2]
            )
        )

    def render(self):
        return self.env.render()

class FlattenActionWrapperDilation(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        print(env.action_space)
        self.action_space = spaces.MultiDiscrete([
            env.action_space[0].nvec[0], # kernel 0
            env.action_space[0].nvec[1], # kernel 1
            env.action_space[1].n, # job
            env.action_space[2].n, # should skip time
            env.action_space[3].n  # should zoom-in
        ])

    def action(self, action: np.ndarray):
        final_action = DilationEnvironmentAction(
            selected_machine_cell=(
                action[0],
                action[1]
            ),
            selected_job=action[2],
            execute_schedule_command=action[3],
            contract=action[4]
        )
        return final_action

    def render(self):
        return self.env.render()


from gymnasium.wrappers import TimeLimit

class TimeLimitPenaltyWrapper(TimeLimit):
    def __init__(self, env, max_episode_steps: int = 1_000, penalty=-10000.0):
        super().__init__(env, max_episode_steps=max_episode_steps)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if truncated and not terminated:
            reward = self.penalty

        return obs, reward, terminated, truncated, info