#!/usr/bin/env python

from cognitivegame import CognitiveGame
from episode import Episode
import maxent as M
import plot as P
import trajectory  as T
import solver as S
import optimizer as O
import img_utils
import value_iteration as vi


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri

def setup_mdp(initial_state, final_states, n_solution, n_token, n_attempt, n_user_action, timeout, trans_filename):
    """This function initialise the mdp

    Args:
        :initial_state: initial state of the mdp 
        :final_states: final states of the mdp
        :n_solution: n of tokens to place in the correct locations
        :n_token: total number of tokens
        :n_attempt: max attempts for token
        :n_user_action: -2: timeout, -1 max_attempt, 0, wrong move, 1 correct move
        :timeout: max time for user to move a token
        :trans_filename: the transition probabilities generated with the simulator
    Return:
        :word: reference to the mdp
        :rewards
        :terminals vector (index)
    """
    world = CognitiveGame(initial_state, final_states, n_solution, n_token, n_attempt, n_user_action, timeout, trans_filename)

    terminals = list()
    # set up the reward function
    reward = np.zeros(world.n_states)
    for idx, final_state in enumerate(final_states):
      index_final_state = world.state_point_to_index(final_state)
      reward[index_final_state] = 1
      terminals.append(index_final_state)

    return world, reward, terminals


def generate_trajectories(world, reward, terminal):
    """Generate some "expert" trajectories.

    Args:
        :word: reference to the mdp
        :reward: the reward vector
        :terminal: the terminal vector
    Return:
          trajectories according to the mdp model
    """
    # parameters
    n_trajectories = 200
    discount = 0.7
    weighting = lambda x: x**5

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    initial[0] = 1.0
    terminal_states = [world.state_point_to_index(t) for t in terminal]
    # generate trajectories
    value = S.value_iteration(world.p_transition, reward, discount)
    policy = S.stochastic_policy_from_value(world, value, w=weighting)
    policy_exec = T.stochastic_policy_adapter(policy)
    tjs = list(T.generate_trajectories(n_trajectories, world, policy_exec, initial, terminal_states))

    return tjs, policy


def maxent(world, terminal, trajectories):
    """
    Maximum Entropy Inverse Reinforcement Learning

     Args:
        :word: reference to the mdp
        :terminal: the terminal vector
        :trajectories: The trajectories generated with the simulator
    Return:
        estimation of the reward based on the MEIRL
    """
    # set up features: we use one feature vector per state
    features = world.assistive_feature(trajectories)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.1))

    # actually do some inverse reinforcement learning
    reward = M.irl(world.p_transition, features, terminal, trajectories, optim, init)

    return reward


def maxent_causal(world, terminal, trajectories, discount=0.8):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = CognitiveGame.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = M.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount)

    return reward


def main():
    # common style arguments for plotting
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }
    n_solution = 11
    n_token = 15
    n_attempt = 5
    n_user_action = 4

    initial_state = (1, 1, 0)
    final_states = [(n_solution, a, u)   for a in range(1, n_attempt+1) for u in range(-2, 2) ]

    episode_obj = Episode()
    episodes = episode_obj.load_episodes(file="../trajectories/trajectories_generation0.npy", episode=5, population=10)

    # set-up mdp
    world, reward, terminals = setup_mdp(initial_state, final_states, n_solution,
                                         n_token, n_attempt, n_user_action, 15, episodes)

    exp_V, exp_P = vi.value_iteration(world.p_transition, reward, gamma=0.9, error=1e-3, deterministic=True)

    #PLOTS EXPERT
    plt.figure(figsize=(20, 11), num="exp_rew")
    sns.heatmap(np.reshape(reward, (11, 20)), cmap="Spectral", annot=True, cbar=False)
    plt.figure(figsize=(20, 11), num="exp_V")
    sns.heatmap(np.reshape(exp_V, (11, 20)), cmap="Spectral", annot=True, cbar=False)
    plt.figure(figsize=(20, 11), num="exp_P")
    sns.heatmap(np.reshape(exp_P, (11, 20)), cmap="Spectral", annot=True, cbar=False)

    # # maximum entropy reinforcement learning (non-causal)
    maxent_R = maxent(world, terminals, episodes)
    #maxent_V = S.value_iteration(world.p_transition, maxent_R, discount=0.9, eps=1e-3)
    maxent_V, maxent_P = vi.value_iteration(world.p_transition, maxent_R, gamma=0.9, error=1e-5, deterministic=True)

    #PLOTS MAXENT IRL
    plt.figure(figsize=(20, 11), num="maxent_R")
    sns.heatmap(np.reshape(maxent_R, (11, 20)), cmap="Spectral", annot=True, cbar=False)
    plt.figure(figsize=(20, 11), num="maxent_V")
    sns.heatmap(np.reshape(maxent_V, (11, 20)), cmap="Spectral", annot=True, cbar=False)
    plt.figure(figsize=(20, 11), num="maxent_P")
    sns.heatmap(np.reshape(maxent_P, (11, 20)), cmap="Spectral", annot=True, cbar=False)

    # # maximum casal entropy reinforcement learning (non-causal)
    maxcausal_R = maxent_causal(world, terminals, episodes)
    #maxcausal_V = S.value_iteration(world.p_transition, maxcausal_R, discount=0.9, eps=1e-3)
    maxcasual_V, maxcasual_P = vi.value_iteration(world.p_transition, maxent_R, gamma=0.9, error=1e-3,
                                                            deterministic=True)
    # PLOTS MAXENT_CAUSAL IRL
    # plt.figure(figsize=(20, 11), num="maxcausal_R")
    # sns.heatmap(np.reshape(maxent_R, (11, 20)), cmap="Spectral", annot=True, cbar=False)
    # plt.figure(figsize=(20, 11), num="maxcausal_V")
    # sns.heatmap(np.reshape(maxcasual_V, (11, 20)), cmap="Spectral", annot=True, cbar=False)
    # plt.figure(figsize=(20, 11), num="maxcausal_P")
    # sns.heatmap(np.reshape(maxcasual_P, (11, 20)), cmap="Spectral", annot=True, cbar=False)



    #we interact with the user which provided those others trajectories
    new_episodes = episode_obj.load_episodes(file="../trajectories/trajectories_generation0.npy", episode=1,population=10)
    evauation_episodes = episodes+new_episodes
    maxent_R_eval = maxent(world, terminals, evauation_episodes)
    #maxent_V = S.value_iteration(world.p_transition, maxent_R, discount=0.9, eps=1e-3)
    maxent_V_eval, maxent_P_eval = vi.value_iteration(world.p_transition, maxent_R_eval, gamma=0.9, error=1e-5, deterministic=True)

    plt.figure(figsize=(20, 11), num="maxent_R_eval")
    sns.heatmap(np.reshape(maxent_R_eval, (11, 20)), cmap="Spectral", annot=True, cbar=False)
    plt.figure(figsize=(20, 11), num="maxent_V_eval")
    sns.heatmap(np.reshape(maxent_V_eval, (11, 20)), cmap="Spectral", annot=True, cbar=False)
    plt.figure(figsize=(20, 11), num="maxent_P_eval")
    sns.heatmap(np.reshape(maxent_P_eval, (11, 20)), cmap="Spectral", annot=True, cbar=False)
    plt.show()


if __name__ == '__main__':
    main()
