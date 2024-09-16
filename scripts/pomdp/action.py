"""
Defines the Action for the 2D Multi-Object Search domain;

Action space:

    Motion :math:`\cup` Look :math:`\cup` Find

* Motion Actions scheme 1: South, East, West, North.
* Motion Actions scheme 2: Left 45deg, Right 45deg, Forward
* Look: Interprets sensor input as observation
* Find: Marks objects observed in the last Look action as
  (differs from original paper; reduces action space)

It is possible to force "Look" after every N/S/E/W action;
then the Look action could be dropped. This is optional behavior.
"""

import pomdp_py
import math


###### Actions ######
class Action(pomdp_py.Action):
    """Mos action; Simple named action."""

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Action(%s)" % self.name

class TurnAction(Action):
    def __init__(self, turn_left):
        self.turn_left = turn_left
        if turn_left:
            name = "left"
        else:
            name = "right"
        super().__init__("Turn %s" % (name))

class ForwardAction(Action):
    # For simplicity, this LookAction is not parameterized by direction
    def __init__(self, weight):
        self.weight = weight
        super().__init__("Forward %sm" % (self.weight))

class DeclareAction(Action):
    # For simplicity, this LookAction is not parameterized by direction
    def __init__(self):
        super().__init__("Declare")

TurnLeft = TurnAction(True)
TurnRight = TurnAction(False)
Forward = ForwardAction(0.25)
Declare = DeclareAction()

ALL_MOTION_ACTIONS = [TurnLeft, TurnRight, Forward, Declare]

# print(ALL_MOTION_ACTIONS)
# Look = LookAction()
# Find = FindAction()

# if MOTION_SCHEME == "xy":
#     ALL_MOTION_ACTIONS = [MoveEast, MoveWest, MoveNorth, MoveSouth]
# elif MOTION_SCHEME == "vw":
#     ALL_MOTION_ACTIONS = [MoveForward, MoveBackward, MoveLeft, MoveRight]
# else:
#     raise ValueError("motion scheme '%s' is invalid" % MOTION_SCHEME)
