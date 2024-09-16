import numpy as np
import math
import random
import matplotlib.pyplot as plt
from copy import copy, deepcopy
class Map:
  def __init__(self):
    self.state_map_reset = ['SOOOO',
                            'OOOOO', 
                            'OOOOO',
                            'OOOOO',
                            'OOOOG']
    self.default_reset = ['OOOOO',
                            'OOOOO', 
                            'OOOOO',
                          'OOOOO',
                            'OOOOG']
    self.state_map = ['SOOOO',
                      'OOOOO', 
                      'OOOOO',
                      'OOOOO',
                      'OOOOG']
    self.default = ['OOOOO',
                            'OOOOO', 
                            'OOOOO',
                            'OOOOO',
                            'OOOOG'] 
    # 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
    self.action_to_direction = {
            0: np.array([0, -1]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
        }
    self.row = len(self.state_map)
    self.col = len(self.state_map[0])
    self.done = False   
    self.num_steps = 0 

  def find_state(self):
    for i in range(self.row):
      if self.state_map[i].find('S') != -1:
        #print(i)
        #return i*self.row + self.state_map[i].find('S')
        return i*self.col + self.state_map[i].find('S')
    print('Cant find state!')
    return None


  def isValid(self, row, col):
    return row >= 0 and row < self.row and col >= 0 and col < len(self.state_map[0])

  def numActions(self):
    # move actions + inspect
    return len(self.action_to_direction)
  
  def numStates(self):
    return len(self.state_map) * len(self.state_map[1])
  
  def reset(self):
    self.done = False
    self.state_map = deepcopy(self.state_map_reset)
    self.default = deepcopy(self.default_reset) 
    self.num_steps = 0 
    return self.find_state()

  """
  @param: action
  @output: state given action
           reward given previous state and action
           if game done
  """
  def step(self, action):
    # can do action only if not finish
    if not self.done:
      self.num_steps += 1 
      row = math.floor(self.find_state()/self.col) 
      col = self.find_state()%self.col
      r = row
      c = col
      observation = np.full(4, False)
      reward =0
      # if action is a movement action
      if action == 0 or action == 1 or action == 2 or action == 3:
        new_row = row + self.action_to_direction[action][0]
        new_col = col + self.action_to_direction[action][1]
        #print('NEW: ', new_row, new_col)
        # if s', where (s' = s given a), is in range of the map        
        if self.isValid(new_row, new_col):
          # put previous state to what it was
          self.state_map[row] = self.state_map[row][:col] + self.default[row][col] + self.state_map[row][col+1:]
          # new state = S
          self.state_map[new_row] = self.state_map[new_row][:new_col] + 'S' + self.state_map[new_row][new_col+1:]
          
          r = new_row
          c = new_col
          # if exit node
          if self.default[new_row][new_col] == 'G':
            reward = 10#30 - self.num_steps
            self.done = True
            #print('Game over!')
        else:
          reward -= 1

        
      return r*self.col+ c, reward, self.done,
    else:
      #print('You have finished. Please reset!')
      return None

  def print_map(self):
    for i in self.state_map:
      print(i)

total_episodes = 100         # Total episodes
learning_rate = 0.05           # Learning rate
max_steps = 30               # Max steps per episode
gamma = 0.90                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob
rewards = []
step_q = []
rewards_bf4 = []
step_bf4 = []
rewards_bf9 = []
step_bf9 = []
m = Map()
m_bf4 = Map()
bf4 = [6,8,16,18]
bf9 = [6,7,8,11,12,13,16,17,18]

def pose(state):
  # output (y,x) or (row, col)
  return math.floor(state/5), state%5
qtable = np.zeros((m.numStates(), m.numActions()))
# Training
step_q.clear()
rewards.clear()
for episode in range(total_episodes):
    # Reset the environment
    state = m.reset()
    steps = 0
    done = False
    total_rewards = 0
    
    # True:
    # 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
    for steps in range(max_steps):
      # 3. Choose an action a in the current world state (s)
      ## First we randomize a number
      exp_exp_tradeoff = random.uniform(0, 1)
      
      ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
      if exp_exp_tradeoff > epsilon: #or episode > 40:
        action = np.argmax(qtable[state,:])

      # Else doing a random choice --> exploration
      else:
          action = random.randint(0, m.numActions()-1)
      #print("Action: ", action)
      # Take the action (a) and observe the outcome state(s') and reward (r)
      new_state, reward, done = m.step(action)
      qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
      
      total_rewards += reward
      
      # Our new state is state
      state = new_state
      # If done (if we're dead) : finish episode
      if done == True:           
        break
    step_q.append(steps)        
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)

#print ("Score over time: " +  str(sum(rewards)/total_episodes))
#print(qtable)
x = np.arange(0, len(rewards))
plt.plot(x, rewards, color ="green")
plt.show()
plt.plot(x, step_q, color ="green")
plt.show()

# action = -1 means u want all action, 0-3 means just 1 of the action
def phi(state, bf, action = -1, r = 1):
  output = []
  phi_all_action = []
  y,x = pose(state)   
  for i in bf:
    y_bf,x_bf = pose(i)  
    output.append(np.exp(   - (np.linalg.norm(np.array([x,y]) - np.array([x_bf, y_bf]))**2) / (2 * (r**2))  )    )
  #print(output)
  if action == -1:
    for i in range(4):
      phi = np.zeros(len(bf)*4)
      for l in range(len(bf)):
        phi[i*len(bf) +l] = output[l]
      phi_all_action.append(phi)
    return phi_all_action
    # phi_mult_theta = []
    # for phi in phi_all_action:
    #   phi_mult_theta.append(phi@theta)
    #max_action_keys = [jj for jj, j in enumerate(phi_mult_theta) if j == max(phi_mult_theta)]
    print(phi_all_action)
  else:
    phi = np.zeros(len(bf)*4)
    for l in range(len(bf)):
        phi[action*len(bf) +l] = output[l]
    return phi
step_bf4.clear()
rewards_bf4.clear()
theta = np.zeros(16)
#theta = np.zeros([4,4])
#theta = 0
for episode in range(100):
    # Reset the environment
    state = m_bf4.reset()
    steps = 0
    done = False
    total_rewards = 0
    
    # True:
    # 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
    for steps in range(25):
      # 3. Choose an action a in the current world state (s)
      ## First we randomize a number
      exp_exp_tradeoff = random.uniform(0, 1)
      
      ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
      if exp_exp_tradeoff > 0.02: #or episode > 40:
        phi_mult_theta = []
        for ph in phi(state,bf4):
          phi_mult_theta.append(ph@theta)
        action = np.argmax(phi_mult_theta)
      # Else doing a random choice --> exploration
      else:
        action = random.randint(0, m_bf4.numActions()-1)
      #print("Action: ", action)
      # Take the action (a) and observe the outcome state(s') and reward (r)
      new_state, reward, done = m_bf4.step(action)
      all_action = []
      p = phi(new_state,bf4)
      for i in p:
          all_action.append(i@theta)
      theta = theta + learning_rate * (reward + gamma * np.max(all_action) - (phi(state,bf4, action) @ theta)) * phi(state,bf4, action)
      #theta[action] = theta[action] + learning_rate * (reward + gamma * np.max(phi(new_state, bf4).T @ theta) - (phi(state, bf4).T @ theta)[0,action]) * phi(state, bf4)[:,0]
      #theta = theta + learning_rate * (reward + gamma * np.max(phi(new_state, bf4) * theta) - (phi(state, bf4)[action] * theta) ) * (phi(state, bf4)[action])
      total_rewards += reward
      
      # Our new state is state
      state = new_state
      # If done (if we're dead) : finish episode
      if done == True:           
        break
    step_bf4.append(steps)        
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards_bf4.append(total_rewards)

#print ("Score over time: " +  str(sum(rewards)/total_episodes))
#print(qtable)
x = np.arange(0, len(rewards_bf4))
plt.plot(x, rewards_bf4, label='bf4')
plt.plot(x, rewards, label='QL')
plt.title('Reward')
leg = plt.legend(loc='upper right')
plt.show()
plt.plot(x, step_bf4, label='bf4')
plt.plot(x, step_q, label='QL')
plt.title('Steps')
leg = plt.legend(loc='upper right')
plt.show()