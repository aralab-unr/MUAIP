import pomdp_py
from state import *
from transition import *
from reward_model import *

class CAISEnvironment(pomdp_py.Environment):
    """"""

    def __init__(self, min_max_dim, init_state, sensors, robot_position_states):
        """
        Args:
            sensors (dict): Map from robot_id to sensor (Sensor);
                            Sensors equipped on robots; Used to determine
                            which objects should be marked as found.
            min_max_dim (np.array): 2D boundary of robot states
                            [ [min_x, max_x], 
                              [min_y, max_y] ]                  
        """
        self.sensors = sensors
        # TODO: transition model
        transition_model = CAISTransitionModel(
            min_max_dim, sensors, set(init_state.object_states.keys()), robot_position_states
        )
        # Target objects, a set of ids, are not robot nor obstacles
        self.target_objects = {
            defid
            for defid in set(init_state.object_states.keys())
            if not isinstance(init_state.object_states[defid], RobotState)
        }
        reward_model = GoalRewardModel(self.target_objects)
        super().__init__(init_state, transition_model, reward_model)

    @property
    def robot_ids(self):
        return set(self.sensors.keys())

    def state_transition(self, action, execute=True, robot_id=None):
        """state_transition(self, action, execute=True, **kwargs)

        Overriding parent class function.
        Simulates a state transition given `action`. If `execute` is set to True,
        then the resulting state will be the new current state of the environment.

        Args:
            action (Action): action that triggers the state transition
            execute (bool): If True, the resulting state of the transition will
                            become the current state.

        Returns:
            float or tuple: reward as a result of `action` and state
            transition, if `execute` is True (next_state, reward) if `execute`
            is False.

        """
        assert (
            robot_id is not None
        ), "state transition should happen for a specific robot"

        next_state = copy.deepcopy(self.state)
        next_state.object_states[robot_id] = self.transition_model[robot_id].sample(
            self.state, action
        )

        reward = self.reward_model.sample(
            self.state, action, next_state, robot_id=robot_id
        )
        # print(reward)
        if execute:
            self.apply_transition(next_state)
            return reward
        else:
            return next_state, reward