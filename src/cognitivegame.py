# """
# Cognitive-Game Markov Decision Processes (MDPs).
#
# Some general remarks:
#     - The state of the agent is defined by game_progress, number of attempt and
#     whether or not the previous move of the user was successfully
#
#     - Any action can be taken in any state and have a unique inteded outcome.
#     The result of an action is stochastic, but there is always exactly one that can be described
#     as the intended result of the action.
# """
#
# import numpy as np
# import itertools
# import random
# import Episode as Episode
#
#
# class CognitiveGame:
#   """
#   Basic deterministic cognitive game MDP.
#
#   The attribute size specifies both width and height of the game, so a
#   game will have size**2 states.
#
#   Args:
#       size: size of the game as integer.
#
#   Attributes:
#       n_states: The number of states of this MDP.
#       n_attempt: the number of attempt for each token
#       n_token: the number of tokens available on the board
#       n_solution: the number of tokens to correctly place on the board
#       n_actions: The number of actions of this MDP.
#       actions: ["Lev_0", "Lev_1","Lev_2","Lev_3","Lev_4"]
#       p_transition: The transition probabilities as table. The entry
#           `p_transition[from, to, a]` contains the probability of
#           transitioning from state `from` to state `to` via action `a`.
#   """
#
#   def __init__(self, initial_state, goal_states, task_length, task_complexity,
#                n_max_attempt, action_space, state_tuple,
#                n_user_action, timeout, trans_matrix):
#     self.n_states = (task_complexity+1)*n_max_attempt*n_user_action
#     self.task_length = task_length
#     self.n_max_attempt = n_max_attempt
#     self.task_complexity = task_complexity
#     self.n_correct_moves = 0
#     self.actions = action_space
#     self.n_user_action = n_user_action
#     self.n_actions = len(self.actions)
#     self.timeout = timeout
#     self.state_tuple = state_tuple#self.generate_states(user_progress=range(1, len(self.task_complexity)+1), user_attempt=range(1, n_max_attempt+1), user_action=range(-1, 2) )
#     self.state_index_tuple_indexed = self.get_states_index(self.state_tuple)
#     self.initial_state_indexed = self.state_point_to_index(initial_state)
#     self.terminal_states_indexed = [self.state_point_to_index(state=final_state) for final_state in (goal_states)]
#     self.rewards = self.initialise_rewards
#
#     #self.p_transition = self.gen_transition_matrix(episodes, self.states_index_tuple, self.actions, self.terminal_states)
#     #self.p_transition = self.load_transition_prob(trans_filename, final_states)
#     self.p_transition = trans_matrix
#
#   def initialise_rewards(self):
#     rewards = np.zeros((len(self.task_complexity), self.n_max_attempt, len(self.user_actions_state)))
#     for t in range(len(self.task_complexity)):
#       for a in range(len(self.attempt)):
#         for u in range(len(self.user_actions_state)):
#           if (t, a, u) in self.goal_states:
#             rewards[t, a, u] = 1
#
#
#   def get_states_index(self, states_space):
#     states_index_list = [self.state_point_to_index(s) for s in (states_space)]
#     return states_index_list
#
#
#   def state_index_to_point(self, index):
#     """
#     Convert a state index to the coordinate representing it.
#
#     Args:
#         state: Integer representing the state.
#
#     Returns:
#         The coordinate as tuple of integers representing the same state
#         as the index.
#     """
#
#     return self.state_tuple[index]
#
#   def state_point_to_index(self, state):
#     """
#     Convert a state coordinate to the index representing it.
#
#     Note:
#         Does not check if coordinates lie outside of the world.
#
#     Args:
#         state: Tuple of integers representing the state.
#
#     Returns:
#         The index as integer representing the same state as the given
#         coordinate.
#     """
#     return self.state_tuple.index(tuple(state))
#
#   def generate_states(self, user_progress, user_attempt, user_action):
#
#     self.state_tuple = list(itertools.product(user_progress, user_attempt, user_action))
#     return self.state_tuple
#
#
#
#   def state_index_transition(self, s, a):
#     """
#     Perform action `a` at state `s` and return the intended next state.
#
#     Does not take into account the transition probabilities. Instead it
#     just returns the intended outcome of the given action taken at the
#     given state, i.e. the outcome in case the action succeeds.
#
#     Args:
#         s: The state at which the action should be taken.
#         a: The action that should be taken.
#
#     Returns:
#         The next state as implied by the given action and state.
#     """
#
#     #get probability for a given state
#     prob_next_states = self.p_transition[s, :, a]
#
#
#     s_point = self.state_index_to_point(s)
#     s_next = 0
#     rand_prob = random.random()
#
#     if s in self.final_states:
#       return s
#
#     s_next = np.argmax(prob_next_states)
#
#
#
#     #s = s[0] + self.actions[a][0], s[1] + self.actions[a][1]
#     return (s_next)
#
#   def gen_transition_matrix(self, episodes, state_list, action_space, terminal_states):
#     '''
#     This function generate the transition matrix function
#     Args:
#         :episodes_path:
#         :n_episode:
#         :n_sol_per_pos:
#         :environment:
#         :path_trans_matrix_occ:
#         :param path_trans_matrix_prob:
#     Return:
#     '''
#     trans_matrix_occ = Episode.generate_statistics(state_list, action_space, episodes)
#     # save the episode on a file
#     # with open(path_trans_matrix_occ, "ab") as f:
#     #   np.save(f, trans_matrix)
#     #   f.close()
#     #trans_matrix_occ = Episode.read_trans_matrix(path_trans_matrix_occ)
#     trans_matrix_prob = Episode.compute_probabilities(trans_matrix_occ, terminal_states)
#     # save the episode on a file
#     # with open(path_trans_matrix_prob, "ab") as f:
#     #   np.save(f, trans_matrix_prob)
#     #   f.close()
#     return trans_matrix_prob
#
#
#   def _transition_prob_table(self):
#     """
#     Builds the internal probability transition table.
#
#     Returns:
#         The probability transition table of the form
#
#             [state_from, state_to, action]
#
#         containing all transition probabilities. The individual
#         transition probabilities are defined by `self._transition_prob'.
#     """
#     table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))
#
#     s1, s2, a = range(self.n_states), range(self.n_states), range(self.n_actions)
#     for s_from in s1:
#       for act in a:
#         for s_to in s2:
#           table[s_from, s_to, act] = self._transition_prob(s_from, s_to, act, 0)
#
#     for state in self.terminal_states_indexed:
#       table[state][state][0] = 1
#     #for s_from, s_to, a in itertools.product(s1, s2, a):
#     #  table[s_from, s_to, a] = self._transition_prob(s_from, s_to, a, 0)
#
#
#     return table
#
#
#   def load_transition_prob(self, file, terminal_states):
#     """
#     Load the transition matrix from a file
#     Args:
#       file: The npy file where the transition prob has been saved
#     Returns:
#        the transition probabily matrix
#     """
#     print("Loading file ...")
#     table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))
#     with open(file, "rb") as f:
#       table = np.load(file, allow_pickle="True")
#
#     s1, s2, a = range(self.n_states), range(self.n_states), range(self.n_actions)
#     for s_from in s1:
#       for act in a:
#         for s_to in s2:
#           if np.isnan(table[s_from, s_to, act]):
#             table[s_from, s_to, act] = 0
#
#     for state in terminal_states:
#       point_to_index = self.state_point_to_index(state)
#       table[point_to_index][point_to_index][0] = 1
#
#     return table
#
#
#   def _transition_prob(self, s_from, s_to, a, value):
#     """
#     Compute the transition probability for a single transition.
#
#     Args:
#         s_from: The state in which the transition originates.
#         s_to: The target-state of the transition.
#         a: The action via which the target state should be reached.
#
#     Returns:
#         The transition probability from `s_from` to `s_to` when taking
#         action `a`.
#     """
#     fx, fy = self.state_index_to_point(s_from)
#     tx, ty = self.state_index_to_point(s_to)
#     lev = (self.actions[a])
#     index_lev = self.actions.index(lev)
#     states_actions = 3*[0]
#
#     max_attempt_states = [(i, self.n_attempt)  for i in range(1, self.n_solution+1)]
#
#
#     if (fx, fy) in max_attempt_states:
#       next_states = [(fx + 1, 1)]
#     elif s_from in self.final_states:
#       next_states = [(fx, fy + 1), (fx, fy)]
#     else:
#       next_states = [(fx + 1, 1), (fx, fy + 1), (fx, fy)]
#
#
#
#     if (fx, fy) in max_attempt_states and s_from in self.final_states:
#       return 0.0
#
#     elif self.state_index_to_point(s_to) not in next_states:
#       return 0.0
#
#     elif (fx, fy) in max_attempt_states and (tx==fx+1 and ty == fy):
#       return 1.0
#
#     print("prev_state:", tx,ty)
#
#     sum = 0
#     prob = list()
#     actions = [0.1, 0.3, 0.5, 0.8, 10]
#     game_timeline = [1, 1.2, 1.5, 2, 2.5]
#     attempt_timeline = [1, 1.2, 1.5, 2]
#     sum_over_actions = 0
#     for next_state in next_states:
#       prob.append(actions[a] * game_timeline[next_state[0]-1] * attempt_timeline[next_state[1]-1])
#       sum_over_actions += actions[a] * game_timeline[next_state[0]-1] * attempt_timeline[next_state[1]-1]
#     norm_prob = list(map(lambda x: x / sum_over_actions, prob))
#     i = 0
#     for ns in next_states:
#       states_actions[i] = (norm_prob[i])
#       i += 1
#
#     if len(next_states) == 3:
#       if tx ==fx + 1 and ty== 1:
#         return states_actions[0]
#       elif tx == fx and ty == fy + 1:
#         return states_actions[1]
#       elif tx == fx and ty == fy:
#         return states_actions[2]
#       else:
#         return 0
#     elif  len(next_states) == 2:
#       if tx == fx and ty == fy + 1:
#         return states_actions[0]
#       elif tx == fx and ty == fy:
#         return states_actions[1]
#       else:
#         return 0
#     else:
#       return 1.0
#
#
#
#   def state_features(self):
#     """
#     Return the feature matrix assigning each state with an individual
#     feature (i.e. an identity matrix of size n_states * n_states).
#
#     Rows represent individual states, columns the feature entries.
#
#     Args:
#         world: A GridWorld instance for which the feature-matrix should be
#             computed.
#
#     Returns:
#         The coordinate-feature-matrix for the specified world.
#     """
#     return np.identity(self.n_states)
#
#   def assistive_feature(self, trajs):
#     """
#     Generate a Nx3 feature map for gridword 1:distance from start state, distance from goal, react_time
#     :param gw:  GridWord
#     :param trajs: generated by the expert
#     :return: Nx3 feature map
#     """
#     max_attempt = self.n_solution*self.n_attempt
#     max_time = self.timeout*self.n_solution*self.n_attempt
#     N = self.n_solution * self.n_attempt * self.n_user_action
#
#
#     feat = np.ones([N, 4])
#     feat_trajs = np.zeros([N, 4])
#     t = 0
#     for traj in trajs:
#       for s1, a, s2 in traj._t:
#         ix, iy, iz = self.state_index_to_point(s1)
#         feat[s1, 0] = 0
#         feat[s1, 1] = 0#max_attempt*ix
#         feat[s1, 2] = ix
#         feat[s1, 3] = iy
#
#         # for trans in traj:
#         #   if trans[0] == i:
#         #     current_react_time = trans[2]
#         #     feat[i, 2] = current_react_time
#       feat_trajs += (feat)
#       t += 1
#     return feat_trajs / len(trajs)
#
#   def coordinate_features(self, world):
#     """
#     Symmetric features assigning each state a vector where the respective
#     coordinate indices are nonzero (i.e. a matrix of size n_states *
#     world_size).
#
#     Rows represent individual states, columns the feature entries.
#
#     Args:
#         world: A GridWorld instance for which the feature-matrix should be
#             computed.
#
#     Returns:
#         The coordinate-feature-matrix for the specified world.
#     """
#     features = np.zeros((world.n_states, world.n_solution))
#
#     for s in range(world.n_states):
#       x, y, _= world.state_index_to_point(s)
#       features[s, x-1] += 1
#       features[s, y-1] += 1
#
#     return features
#
