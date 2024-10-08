"""Policy model for 2D Multi-Object Search domain.
It is optional for the agent to be equipped with an occupancy
grid map of the environment.
"""

import pomdp_py
import random
from action import *


class PolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model. All actions are possible at any state."""

    def __init__(self, robot_id, grid_map=None):
        """FindAction can only be taken after LookAction"""
        self.robot_id = robot_id
        self._grid_map = grid_map

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]

    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplementedError

    def get_all_actions(self, state=None, history=None):
        """note: find can only happen after look."""
        return ALL_MOTION_ACTIONS

    def rollout(self, state=None, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]