"""Reward model for 2D Multi-object Search domain"""

import pomdp_py
from action import *


class CAISRewardModel(pomdp_py.RewardModel):
    def __init__(self, target_objects, big=1000, small=1, robot_id=None):
        """
        robot_id (int): This model is the reward for one agent (i.e. robot),
                        If None, then this model could be for the environment.
        target_objects (set): a set of objids for target objects.
        """
        self._robot_id = robot_id
        self.big = big
        self.small = small
        self._target_objects = target_objects

    def probability(
        self, reward, state, action, next_state, normalized=False, **kwargs
    ):
        if reward == self._reward_func(state, action):
            return 1.0
        else:
            return 0.0

    def sample(self, state, action, next_state, normalized=False, robot_id=None):
        # deterministic
        return self._reward_func(state, action, next_state, robot_id=robot_id)

    def argmax(self, state, action, next_state, normalized=False, robot_id=None):
        """Returns the most likely reward"""
        return self._reward_func(state, action, next_state, robot_id=robot_id)


class GoalRewardModel(CAISRewardModel):
    """
    This is a reward where the agent gets reward only for detect-related actions.
    """

    def _reward_func(self, state, action, next_state, robot_id=None):
        if robot_id is None:
            assert (
                self._robot_id is not None
            ), "Reward must be computed with respect to one robot."
            robot_id = self._robot_id

        reward = 0

        # If the robot has detected all objects
        if len(state.object_states[robot_id]["objects_found"]) == len(
            self._target_objects
        ):
            return 0  # no reward or penalty; the task is finished.

        if isinstance(action, TurnAction):
            reward = reward - self.small 
        elif isinstance(action, ForwardAction):
            reward = reward - (self.small * 3)
        elif isinstance(action, DeclareAction):
            # transition function should've taken care of the detection.
            new_objects_count = len(
                set(next_state.object_states[robot_id].objects_found)
                - set(state.object_states[robot_id].objects_found)
            )
            if new_objects_count == 0:
                # No new detection. "detect" is a bad action.
                reward -= self.big
            else:
                # Has new detection. Award.
                reward += (self.big*new_objects_count)
        return reward
