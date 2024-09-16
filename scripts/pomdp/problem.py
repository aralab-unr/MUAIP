import pomdp_py
from sensor_model import *
from state import *
from env import *
from observation import *
from observation_model import *
from agent import *
import argparse
import time
import random

def convert(robot_id, robot_init_pose, robot_position_states, defects_idx, tunnel_states, sensor):
  robots = {} # robot_id -> robot_state(pose)
  defects = {} # defid -> def_state(pose)
  sensors = {}

  # Calculate distances from robot to each state
  distances = np.linalg.norm(robot_position_states - robot_init_pose[:2], axis=1)
  # Find the index of the closest state
  closest_index = np.argmin(distances)
  # Update robot's (x, y, z) to the closest state's (x, y, z) while retaining its yaw (theta)
  robot_init_pose[:2] = robot_position_states[closest_index]
  robots[robot_id] = RobotState(robot_id, (tuple(robot_init_pose)), (), None)
  # print(robots)
  for idx in defects_idx:
    defid = 1000 + len(defects)
    defects[defid] = ObjectState(defid, "target", tuple( ( tunnel_states)[idx]))
  # print(defects)
  sensors[robot_id] = sensor
  if len(robots) == 0:
    raise ValueError("No initial robot pose!")
  if len(defects) == 0:
      raise ValueError("No object!")
  return robots, defects, sensors


class CAISPOMDP(pomdp_py.OOPOMDP):
  def __init__(self, robot_id, robot_init_pose, targets_idx, robot_position_states, tunnel_states, sensor, sigma=0.01, epsilon=1, belief_rep="histogram", prior={}, num_particles=100):
    """
    robot_init_pose: (x,y,z,theta)
    """
    robots, defects, sensors = convert(robot_id, robot_init_pose, robot_position_states, targets_idx, tunnel_states, sensor)
    init_state = MosOOState({**defects, **robots})
    
    min_x = np.min(robot_position_states[:, 0])
    max_x = np.max(robot_position_states[:, 0])
    min_y = np.min(robot_position_states[:, 1])
    max_y = np.max(robot_position_states[:, 1])

    min_max_dim = np.array([[min_x, max_x], [min_y, max_y]])
    env = CAISEnvironment(min_max_dim, init_state, sensors, robot_position_states)
    observation_model = MosObservationModel(
            min_max_dim, sensor, env.target_objects, sigma=0.01, epsilon=1
        )
    transition_model = CAISTransitionModel(
          min_max_dim, {robot_id: sensor}, env.target_objects, robot_position_states
      )
    agent = MosAgent(
      robot_id,
      env.state.object_states[robot_id],
      env.target_objects,
      tunnel_states,
      env.sensors[robot_id],
      observation_model=observation_model,
      transition_model = transition_model,
      sigma=sigma,
      epsilon=epsilon,
      belief_rep=belief_rep,
      prior=prior,
      num_particles=num_particles,
      grid_map=None,
    )
    print(env.state)

    # o = observation_model.sample(env.state, DeclareAction())
    # print("Observations: ", o)    
    # O0 = ObjectObservationModel(
    #         1000, sensor, min_max_dim, sigma=0.01, epsilon=0.7
    #     )
    # z0 = O0.sample(env.state, DeclareAction())
    # print(z0.pose)
    # print(O0.probability(z0, env.state, DeclareAction()))
    # env.state_transition(TurnAction(True), robot_id=robot_id)
    # print(env.state.object_states[robot_id])
    # env.state_transition(ForwardAction(0.25), robot_id=robot_id)
    # print(env.state.object_states[robot_id])
    # env.state_transition(TurnAction(False), robot_id=robot_id)
    # print(env.state.object_states[robot_id])
    # env.state_transition(ForwardAction(0.25), robot_id=robot_id)
    # env.state_transition(ForwardAction(0.25), robot_id=robot_id)
    # env.state_transition(ForwardAction(0.25), robot_id=robot_id)
    # env.state_transition(ForwardAction(0.25), robot_id=robot_id)
    # env.state_transition(ForwardAction(0.25), robot_id=robot_id)
    # env.state_transition(ForwardAction(0.25), robot_id=robot_id)
    # print(env.state.object_states[robot_id])
    # env.state_transition(DeclareAction(), robot_id=robot_id)
    # print(env.state.object_states[robot_id])
    super().__init__(
        agent,
        env,
        name="MOS(%d)" % (len(env.target_objects)),
    ) 

### Belief Update ###
def belief_update(agent, real_action, real_observation, next_robot_state, planner):
    """Updates the agent's belief; The belief update may happen
    through planner update (e.g. when planner is POMCP)."""
    # Updates the planner; In case of POMCP, agent's belief is also updated.
    planner.update(agent, real_action, real_observation)

    # Update agent's belief, when planner is not POMCP
    if not isinstance(planner, pomdp_py.POMCP):
        # Update belief for every object
        for objid in agent.cur_belief.object_beliefs:
            belief_obj = agent.cur_belief.object_belief(objid)
            if isinstance(belief_obj, pomdp_py.Histogram):
                if objid == agent.robot_id:
                    # Assuming the agent can observe its own state:
                    new_belief = pomdp_py.Histogram({next_robot_state: 1.0})
                else:
                    new_belief = pomdp_py.update_histogram_belief(
                        belief_obj,
                        real_action,
                        real_observation.for_obj(objid),
                        agent.observation_model[objid],
                        agent.transition_model[objid],
                        # The agent knows the objects are static.
                        static_transition=objid != agent.robot_id,
                        oargs={"next_robot_state": next_robot_state},
                    )
            else:
                raise ValueError(
                    "Unexpected program state."
                    "Are you using the appropriate belief representation?"
                )

            agent.cur_belief.set_object_belief(objid, new_belief)

def solve(
    problem,
    max_depth=30,  # planning horizon
    discount_factor=0.99,
    planning_time=1.0,  # amount of time (s) to plan each step
    exploration_const=1000,  # exploration constant
    visualize=True,
    max_time=12000,  # maximum amount of time allowed to solve the problem
    max_steps=500,
):  # maximum number of planning steps the agent can take.  
  random_objid = random.sample(sorted(problem.env.target_objects), 1)[0]
  random_object_belief = problem.agent.belief.object_beliefs[random_objid]
  if isinstance(random_object_belief, pomdp_py.Histogram):
      # Use POUCT
      planner = pomdp_py.POUCT(
          max_depth=max_depth,
          discount_factor=discount_factor,
          # planning_time=planning_time,
          num_sims=500,
          exploration_const=exploration_const,
          rollout_policy=problem.agent.policy_model,
          show_progress=True,
      )  # Random by default 
  else:
    raise ValueError(
        "Unsupported object belief type %s" % str(type(random_object_belief))
    )
  pose_tracking = []
  robot_id = problem.agent.robot_id
  object_poses = []
  if visualize:
    robot_pose = problem.env.state.object_states[robot_id].pose
    for i in problem.env.state.object_states:
      pose = problem.env.state.object_states[i].pose
      if i < 0: 
        color = "red" 
        plt.quiver(pose[0], pose[1], np.cos(pose[2]), np.sin(pose[2]), angles='xy', scale_units='xy', scale=2, color=color)
      else:
        color = "blue"
      plt.scatter(pose[0], pose[1], color=color)
    # for i in problem.agent.cur_belief.object_beliefs:
    #   my_dict = problem.agent.cur_belief.object_beliefs[i].get_histogram()
    #   key_with_max_value = max(my_dict.keys(), key=my_dict.get)
    #   print("Belief Max: ", i, key_with_max_value, my_dict[key_with_max_value])
    # for i in my_dict:
    #   print(my_dict[i])
    plt.savefig(f'{0}.png')
    plt.close()
    # plt.show()
  _time_used = 0
  _find_actions_count = 0
  _total_reward = 0  # total, undiscounted reward
  for i in range(max_steps):
    # Plan action
    _start = time.time()
    real_action = planner.plan(problem.agent)
    _time_used += time.time() - _start
    if _time_used > max_time:
        print("Max time reached")
        break  # no more time to update.
    pose_tracking.append(problem.env.state.object_states[robot_id].pose)
    # print(pose_tracking)
    # Execute action
    reward = problem.env.state_transition(
        real_action, execute=True, robot_id=robot_id
    )

    # Receive observation
    _start = time.time()
    real_observation = problem.env.provide_observation(
        problem.agent.observation_model, real_action
    )

    # Updates
    problem.agent.clear_history()  # truncate history
    problem.agent.update_history(real_action, real_observation)
    belief_update(
        problem.agent,
        real_action,
        real_observation,
        problem.env.state.object_states[robot_id],
        planner,
    )
    _time_used += time.time() - _start

    # Info and render
    _total_reward += reward
    if isinstance(real_action, DeclareAction):
        _find_actions_count += 1
    print("==== Step %d ====" % (i + 1))
    print("Action: %s" % str(real_action))
    print("Observation: %s" % str(real_observation))
    print("Reward: %s" % str(reward))
    print("Reward (Cumulative): %s" % str(_total_reward))
    print("Find Actions Count: %d" % _find_actions_count)
    if isinstance(planner, pomdp_py.POUCT):
        print("__num_sims__: %d" % planner.last_num_sims)
    # Termination check
    if (
        set(problem.env.state.object_states[robot_id].objects_found)
        == problem.env.target_objects
    ):
        print("Done!")
        break
    if _find_actions_count >= len(problem.env.target_objects):
        print("FindAction limit reached.")
        break
    if _time_used > max_time:
        print("Maximum time reached.")
        break
    if visualize:
      print("lol")
      robot_pose = problem.env.state.object_states[robot_id].pose
      for id in problem.env.state.object_states:
        pose = problem.env.state.object_states[id].pose
        if id < 0: 
          color = "red" 
          plt.quiver(pose[0], pose[1], np.cos(pose[2]), np.sin(pose[2]), angles='xy', scale_units='xy', scale=2, color=color)
        else:
          color = "blue"
        plt.scatter(pose[0], pose[1], color=color)
      x_values, y_values,_ = zip(*pose_tracking)
      plt.plot(x_values, y_values, linestyle='-', color='red')
      # for i in problem.agent.cur_belief.object_beliefs:
      #   if i <0:
      #     continue
      #   my_dict = problem.agent.cur_belief.object_beliefs[i].get_histogram()
      #   key_with_max_value = max(my_dict.keys(), key=my_dict.get)
      #   print("Belief Max: ", i, key_with_max_value, my_dict[key_with_max_value], problem.env.state.object_states[i])
      # print(problem.agent.cur_belief.random())
      plt.savefig(f'{i+1}.png')
      plt.close()
      # plt.show()
  pose_tracking.append(problem.env.state.object_states[robot_id].pose)
  with open('output.txt', 'w') as file:
      for item in pose_tracking:
          # Write each tuple to the file, converting it to a string
          file.write(f"{item}\n")
  print(pose_tracking)
# FOV param
h_fov = 61
v_fov = 49
near = 0.2
far = 3.0
tf0 = np.array([0.155, 0.028, 0.313, 0.0, 0.0, 0.0])
tf1 = np.array([0.084, 0.143, 0.313, 0.0, 0.0, 0.8])
tf2 = np.array([0.125, -0.104, 0.313,0.0, 0.0, -0.8])

# tunnel states params
w = 1.2
l = 4.0
h = 0.7
scale = 5.5
offset_x = 0.0
offset_y = -0.6 + 0.013
offset_z = 0.0

# Robot states param
robot_l = 1.0
robot_offset_x = offset_x-1.0
robot_offset_y = offset_y+0.287
robot_offset_z = offset_z
robot_y_scale = 7.0

# declare
sensor = MultiSensor(-1, h_fov, v_fov, near, far, tf0, tf1, tf2)
tunnel_states = generate_uniform_grid(w, l, h, offset_x=offset_x, offset_y=offset_y, offset_z=offset_z, scale=scale)
defects_idx = np.array([23,24,71,80,111])#np.array([31,45,86,92,97])
robot_init_pose = np.array([-0.75, 0.0, 0.0])#np.array([-0.75, 0.0, 0.0, 0.0])
robot_position_states = generate_robot_state(w/2, l+robot_l, offset_x=robot_offset_x, offset_y=robot_offset_y, offset_z=robot_offset_z, scale_x=scale, scale_y=robot_y_scale)
problem = CAISPOMDP(-1, robot_init_pose, defects_idx, robot_position_states, tunnel_states, sensor)
# print(problem.agent.cur_belief.random())
# print(problem.agent.cur_belief.random())
print(robot_position_states.shape)
solve(
  problem,
  max_depth=15,
  discount_factor=0.99,
  planning_time=10.0,
  exploration_const=1000,
  visualize=True,
  max_time=43200,
  max_steps=500,
    )