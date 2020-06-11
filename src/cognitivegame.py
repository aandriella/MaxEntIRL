"""
Cognitive-Game Markov Decision Processes (MDPs).

Some general remarks:
    - The state of the agent is defined by game_progress, number of attempt and
    whether or not the previous move of the user was successfully

    - Any action can be taken in any state and have a unique inteded outcome.
    The result of an action is stochastic, but there is always exactly one that can be described
    as the intended result of the action.
"""

import numpy as np
import itertools
import random


class CognitiveGame:
  """
  Basic deterministic cognitive game MDP.

  The attribute size specifies both width and height of the game, so a
  game will have size**2 states.

  Args:
      size: size of the game as integer.

  Attributes:
      n_states: The number of states of this MDP.
      n_attempt: the number of attempt for each token
      n_token: the number of tokens available on the board
      n_solution: the number of tokens to correctly place on the board
      n_actions: The number of actions of this MDP.
      actions: ["Lev_0", "Lev_1","Lev_2","Lev_3","Lev_4"]
      p_transition: The transition probabilities as table. The entry
          `p_transition[from, to, a]` contains the probability of
          transitioning from state `from` to state `to` via action `a`.
  """

  def __init__(self, initial_state, final_states, n_solution, n_token, n_attempt, n_user_action, trans_filename):
    self.n_solution = n_solution
    self.n_token = n_token
    self.n_attempt = n_attempt
    self.actions = [0, 1, 2, 3, 4]
    self.n_user_action = n_user_action
    self.n_states = n_solution * n_attempt * n_user_action
    self.n_actions = len(self.actions)
    self.states_tuple = self.generate_states(user_progress=range(1, self.n_solution+1), user_attempt=range(1, n_attempt+1), user_action=range(-2, 2) )
    self.initial_state = self.state_point_to_index(initial_state)
    self.final_states = [self.state_point_to_index(state=final_state) for final_state in (final_states)]
    self.p_transition = self.load_transition_prob(trans_filename)


  def state_index_to_point(self, index):
    """
    Convert a state index to the coordinate representing it.

    Args:
        state: Integer representing the state.

    Returns:
        The coordinate as tuple of integers representing the same state
        as the index.
    """

    return self.states_tuple[index]

  def state_point_to_index(self, state):
    """
    Convert a state coordinate to the index representing it.

    Note:
        Does not check if coordinates lie outside of the world.

    Args:
        state: Tuple of integers representing the state.

    Returns:
        The index as integer representing the same state as the given
        coordinate.
    """
    return self.states_tuple.index(tuple(state))

  def generate_states(self, user_progress, user_attempt, user_action):

    self.states_tuple = list(itertools.product(user_progress, user_attempt, user_action))
    return self.states_tuple



  def state_index_transition(self, s, a):
    """
    Perform action `a` at state `s` and return the intended next state.

    Does not take into account the transition probabilities. Instead it
    just returns the intended outcome of the given action taken at the
    given state, i.e. the outcome in case the action succeeds.

    Args:
        s: The state at which the action should be taken.
        a: The action that should be taken.

    Returns:
        The next state as implied by the given action and state.
    """
    s_point = self.state_index_to_point(s)
    s_next = 0

    if s in self.final_states:
      return s

    if s_point[1]>=4:
      s_next = s_point[0]+1, 1
    else:
      if a == 0:
        if random.random()<0.2:
          s_next = s_point[0]+1, 1
        else:
          s_next = s_point[0], s_point[1]+1
        pass
      elif a == 1:
        if random.random()<0.4:
          s_next = s_point[0]+1, 1
        else:
          s_next = s_point[0], s_point[1]+1
      elif a == 2:
        if random.random()<0.6:
          s_next = s_point[0]+1, 1
        else:
          s_next = s_point[0], s_point[1]+1
      elif a == 3:
        if random.random()<0.8:
          s_next = s_point[0]+1, 1
        else:
          s_next = s_point[0], s_point[1]+1
      elif a == 4:
        if random.random()<1.0:
          s_next = s_point[0]+1, 1
        else:
          s_next = s_point[0], s_point[1]+1


    #s = s[0] + self.actions[a][0], s[1] + self.actions[a][1]
    return self.state_point_to_index(s_next)

  def _transition_prob_table(self):
    """
    Builds the internal probability transition table.

    Returns:
        The probability transition table of the form

            [state_from, state_to, action]

        containing all transition probabilities. The individual
        transition probabilities are defined by `self._transition_prob'.
    """
    table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))

    s1, s2, a = range(self.n_states), range(self.n_states), range(self.n_actions)
    for s_from in s1:
      for act in a:
        for s_to in s2:
          table[s_from, s_to, act] = self._transition_prob(s_from, s_to, act, 0)
    #for s_from, s_to, a in itertools.product(s1, s2, a):
    #  table[s_from, s_to, a] = self._transition_prob(s_from, s_to, a, 0)

    return table


  def load_transition_prob(self, file):
    """
    Load the transition matrix from a file
    Args:
      file: The npy file where the transition prob has been saved
    Returns:
       the transition probabily matrix
    """
    print("Loading file ...")
    table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))
    with open(file, "rb") as f:
      table = np.load(file, allow_pickle="True")

    s1, s2, a = range(self.n_states), range(self.n_states), range(self.n_actions)
    for s_from in s1:
      for act in a:
        for s_to in s2:
          if np.isnan(table[s_from, s_to, act]):
            table[s_from, s_to, act] = 0

    return table


  def _transition_prob(self, s_from, s_to, a, value):
    """
    Compute the transition probability for a single transition.

    Args:
        s_from: The state in which the transition originates.
        s_to: The target-state of the transition.
        a: The action via which the target state should be reached.

    Returns:
        The transition probability from `s_from` to `s_to` when taking
        action `a`.
    """
    fx, fy = self.state_index_to_point(s_from)
    tx, ty = self.state_index_to_point(s_to)
    lev = (self.actions[a])
    index_lev = self.actions.index(lev)
    states_actions = 3*[0]

    max_attempt_states = [(i, self.n_attempt)  for i in range(1, self.n_solution+1)]


    if (fx, fy) in max_attempt_states:
      next_states = [(fx + 1, 1)]
    elif s_from in self.final_states:
      next_states = [(fx, fy + 1), (fx, fy)]
    else:
      next_states = [(fx + 1, 1), (fx, fy + 1), (fx, fy)]



    if (fx, fy) in max_attempt_states and s_from in self.final_states:
      return 0.0

    elif self.state_index_to_point(s_to) not in next_states:
      return 0.0

    elif (fx, fy) in max_attempt_states and (tx==fx+1 and ty == fy):
      return 1.0







    print("prev_state:", tx,ty)

    sum = 0
    prob = list()
    actions = [0.1, 0.3, 0.5, 0.8, 10]
    game_timeline = [1, 1.2, 1.5, 2, 2.5]
    attempt_timeline = [1, 1.2, 1.5, 2]
    sum_over_actions = 0
    for next_state in next_states:
      prob.append(actions[a] * game_timeline[next_state[0]-1] * attempt_timeline[next_state[1]-1])
      sum_over_actions += actions[a] * game_timeline[next_state[0]-1] * attempt_timeline[next_state[1]-1]
    norm_prob = list(map(lambda x: x / sum_over_actions, prob))
    i = 0
    for ns in next_states:
      states_actions[i] = (norm_prob[i])
      i += 1

    if len(next_states) == 3:
      if tx ==fx + 1 and ty== 1:
        return states_actions[0]
      elif tx == fx and ty == fy + 1:
        return states_actions[1]
      elif tx == fx and ty == fy:
        return states_actions[2]
      else:
        return 0
    elif  len(next_states) == 2:
      if tx == fx and ty == fy + 1:
        return states_actions[0]
      elif tx == fx and ty == fy:
        return states_actions[1]
      else:
        return 0
    else:
      return 1.0



  def state_features(world):
    """
    Return the feature matrix assigning each state with an individual
    feature (i.e. an identity matrix of size n_states * n_states).

    Rows represent individual states, columns the feature entries.

    Args:
        world: A GridWorld instance for which the feature-matrix should be
            computed.

    Returns:
        The coordinate-feature-matrix for the specified world.
    """
    return np.identity(world.n_states)

  def assistive_feature(self, initial_state, final_states, trajs):
    """
    Generate a Nx3 feature map for gridword 1:distance from start state, distance from goal, react_time
    :param gw:  GridWord
    :param trajs: generated by the expert
    :return: Nx3 feature map
    """
    N = self.n_solution * self.n_attempt
    in_state_x, in_state_y = initial_state
    end_state_x, end_state_y = final_state
    feat = np.zeros([N, 3])
    feat_trajs = np.zeros([N, 3])
    t = 0
    for traj in trajs:
      for i in range(N):
        iy, ix = gw.idx2pos(i)
        feat[i, 0] = abs(ix - in_state_y) - abs(ix - in_state_x)
        feat[i, 1] = abs(ix - end_state_y) - abs(ix - in_state_x)
        for trans in traj:
          if trans[0] == i:
            current_react_time = trans[2]
            feat[i, 2] = current_react_time
      feat_trajs += (feat)
      t += 1
    return feat_trajs / len(trajs)

  def coordinate_features(world):
    """
    Symmetric features assigning each state a vector where the respective
    coordinate indices are nonzero (i.e. a matrix of size n_states *
    world_size).

    Rows represent individual states, columns the feature entries.

    Args:
        world: A GridWorld instance for which the feature-matrix should be
            computed.

    Returns:
        The coordinate-feature-matrix for the specified world.
    """
    features = np.zeros((world.n_states, world.n_solution))

    for s in range(world.n_states):
      x, y, _= world.state_index_to_point(s)
      features[s, x-1] += 1
      features[s, y-1] += 1

    return features



if __name__ == '__main__':
  n_solution = 5
  n_token = 15
  n_attempt = 4
  n_user_action = 3
  cg = CognitiveGame(n_solution, n_token, n_attempt, n_user_action)
  curr_state = cg.state_point_to_index(state=(1,1,1))
  next_state = cg.state_point_to_index(state =(2,1,1))

  t = cg._transition_prob(curr_state, next_state, a=0, value=0)
