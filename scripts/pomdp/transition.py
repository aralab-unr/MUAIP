"""Defines the TransitionModel for the 2D Multi-Object Search domain.

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Description: Multi-Object Search in a 2D grid world.

Transition: deterministic
"""
import numpy as np
import pomdp_py
import copy
from state import *
from observation import *
from action import *


####### Transition Model #######
class CAISTransitionModel(pomdp_py.OOTransitionModel):
    """Object-oriented transition model; The transition model supports the
    multi-robot case, where each robot is equipped with a sensor; The
    multi-robot transition model should be used by the Environment, but
    not necessarily by each robot for planning.
    """

    def __init__(self, min_max_dim, sensors, object_ids, robot_position_states, epsilon=1e-9):
        """
        sensors (dict): robot_id -> Sensor
        for_env (bool): True if this is a robot transition model used by the
             Environment.  see RobotTransitionModel for details.
        """
        self._sensors = sensors
        transition_models = {
            objid: StaticDefectTransitionModel(objid, epsilon=epsilon)
            for objid in object_ids
            if objid not in sensors
        }
        for robot_id in sensors:
            transition_models[robot_id] = RobotTransitionModel(
                sensors[robot_id], min_max_dim, robot_position_states, epsilon=epsilon
            )
        super().__init__(transition_models)

    def sample(self, state, action, **kwargs):
        oostate = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs)
        return MosOOState(oostate.object_states)

    def argmax(self, state, action, normalized=False, **kwargs):
        oostate = pomdp_py.OOTransitionModel.argmax(self, state, action, **kwargs)
        return MosOOState(oostate.object_states)


class StaticDefectTransitionModel(pomdp_py.TransitionModel):
    """This model assumes the object is static."""

    def __init__(self, objid, epsilon=1e-9):
        self._objid = objid
        self._epsilon = epsilon

    def probability(self, next_object_state, state, action):
        if next_object_state != state.object_states[next_object_state["id"]]:
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def sample(self, state, action):
        """Returns next_object_state"""
        return self.argmax(state, action)

    def argmax(self, state, action):
        """Returns the most likely next object_state"""
        return copy.deepcopy(state.object_states[self._objid])


class RobotTransitionModel(pomdp_py.TransitionModel):
    """We assume that the robot control is perfect and transitions are deterministic."""

    def __init__(self, sensor, min_max_dim, robot_position_states, epsilon=1e-9):
        """
        dim (tuple): a tuple (width, length) for the dimension of the world
        """
        # this is used to determine objects found for FindAction
        self._sensor = sensor
        self._robot_id = sensor.robot_id
        self._min_max_dim = min_max_dim
        self._epsilon = epsilon
        self._robot_position_states = robot_position_states

    @classmethod
    def if_move_by(cls, robot_id, state, action, min_max_dim, robot_position_states, check_collision=True):
        """Defines the dynamics of robot motion;
        dim (tuple): the width, length of the search world."""
        robot_pose = state.pose(robot_id)
        rx, ry, rth = robot_pose
        if isinstance(action, TurnAction):
            if action.turn_left:
                rth += np.deg2rad(45) 
            else:
                rth -= np.deg2rad(45) 
            # Normalize yaw to keep it within the range [-π, π]
            rth = (rth + np.pi) % (2 * np.pi) - np.pi
            # print(rx, ry, rz, rth)
            return (rx, ry, rth)
        elif isinstance(action, ForwardAction):
            weight = action.weight
            rx = (rx + weight * math.cos(rth))
            ry = (ry + weight * math.sin(rth))
            # no movement if out of boundary
            if not in_boundary((rx,ry), min_max_dim):
                return robot_pose
            else:      
                reference_point = np.array([rx, ry])
                # Calculate the Euclidean distance between the reference point and all points in robot_position_states
                distances = np.linalg.norm(robot_position_states - reference_point, axis=1)
                # Find the index of the minimum distance
                closest_index = np.argmin(distances)
                # Return the closest point
                # print(*robot_position_states[closest_index], rth)
                return (*robot_position_states[closest_index], rth)
        else:
            raise ValueError("Cannot move robot with %s action" % str(type(action)))

    def probability(self, next_robot_state, state, action):
        if next_robot_state != self.argmax(state, action):
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def argmax(self, state, action):
        """Returns the most likely next robot_state"""
        if isinstance(state, RobotState):
            robot_state = state
            # not happening breh
        else:
            robot_state = state.object_states[self._robot_id]
            # print(robot_state)

        next_robot_state = copy.deepcopy(robot_state)
        # camera direction is only not None when looking
        next_robot_state["camera_direction"] = None
        if isinstance(action, TurnAction) or isinstance(action, ForwardAction):
            # motion action
            next_robot_state["pose"] = RobotTransitionModel.if_move_by(
                self._robot_id, state, action, self._min_max_dim, self._robot_position_states
            )
        elif isinstance(action, DeclareAction):
            robot_pose = state.pose(self._robot_id)
            # TODO: observe
            z = self._sensor.observe(robot_pose, state)
            # Update "objects_found" set for target objects
            observed_target_objects = {
                objid
                for objid in z.objposes
                if (
                    state.object_states[objid].objclass == "target"
                    and z.objposes[objid] != ObjectObservation.NULL
                )
            }
            # print(next_robot_state["objects_found"] )
            next_robot_state["objects_found"] = tuple(
                set(next_robot_state["objects_found"]) | set(observed_target_objects)
            )
            # print(next_robot_state["objects_found"] )
        return next_robot_state

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)


# Utility functions
def in_boundary(position, boundaries):
    """
    Check if the (x, y) position is out of the defined boundaries.

    Args:
    - position: A tuple or list containing (x, y, z, theta).
    - boundaries: A NumPy array of shape (2, 2) with [[minx, maxx], [miny, maxy]].

    Returns:
    - True if (x, y) is out of bounds, False otherwise.
    """
    x, y = position[0], position[1]
    min_x, max_x = boundaries[0]
    # print(min_x,"<", x, "<", max_x)
    min_y, max_y = boundaries[1]
    # print(min_y,"<", y, "<", max_y)

    # Check if x or y is out of the boundaries
    if x >= min_x and x <= max_x and y >= min_y and y <= max_y:
        return True
    else:
        return False