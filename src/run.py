#!/usr/bin/env python

from cognitivegame import CognitiveGame
from trajectory import Trajectory
import maxent as M
import plot as P
import trajectory  as T
import solver as S
import optimizer as O
import img_utils

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri

def setup_mdp(initial_state, final_states, n_solution, n_token, n_attempt, n_user_action, trans_filename):
    """
    Set-up our MDP/GridWorld
    """
    # create our world

    world = CognitiveGame(initial_state, final_states, n_solution, n_token, n_attempt, n_user_action, trans_filename)

    terminals = list()
    # set up the reward function
    reward = np.zeros(world.n_states)
    for r in range(len(reward)):
      reward[r] = 0
    i=1
    for final_state in (final_states):
      index_final_state = world.state_point_to_index(final_state)
      reward[index_final_state] = 1/i
      terminals.append(index_final_state)
      i+=1

    return world, reward, terminals


def generate_trajectories(world, reward, terminal):
    """
    Generate some "expert" trajectories.
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
    """
    # set up features: we use one feature vector per state
    features = CognitiveGame.state_features(world)

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

    initial_state = (1,1, 0)
    final_states = [(n_solution, a, u) for a in range(1, n_attempt+1) for u in range(-2, 2) ]

    trajectories = T.load_trajectories(file="/home/aandriella/Documents/Codes/IRL/IRLCogniveGame/src/trajectories_generation0.npy", episode=50,population=100)
    print(trajectories)

    # set-up mdp
    world, reward, terminals = setup_mdp(initial_state, final_states, n_solution,
                                         n_token, n_attempt, n_user_action, trans_filename="/home/aandriella/Documents/Codes/IRL/IRLCogniveGame/src/trans_matrix_prob.npy")

    initial_reward = list(tuple())
    for state in range(world.n_states):
        initial_reward.append((world.state_index_to_point(state), reward[state]))
        print("index:",state, " coord:", world.state_index_to_point(state), " reward:", reward[state])



    img_utils.heatmap3d(initial_reward, 'Reward Map - Initial')

    # # show our original reward
    #ax = plt.figure(num='Original Reward').add_subplot(111)
    #p_original_reward = P.plot_state_values(ax, world, reward, title = "Original Reward", **style)
    #
    # # generate "expert" trajectories
    #trajectories, expert_policy = generate_trajectories(world, reward, final_states)
    #trajectories = Trajectory.load_trajectories(file="/home/aandriella/Documents/Codes/IRL/IRLCogniveGame/src/trajectories_generation0.npy", episode=50, population=100)
    #print(trajectories)
    #
    # # show our expert policies
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #P.plot_stochastic_policy(ax, world, expert_policy, **style)
    #
    #for t in trajectories:
    #    P.plot_trajectory(ax, world, t, lw=5, color='blue', alpha=0.025)
    #
    V_origin = S.value_iteration(world.p_transition, reward, discount=0.9, eps=1e-3)
    print(V_origin)
    #
    #
    # # maximum entropy reinforcement learning (non-causal)
    reward_maxent = maxent(world, terminals, trajectories)
    V_maxent = S.value_iteration(world.p_transition, reward_maxent, discount=0.9, eps=1e-3)
    maxent_reward = list(tuple())
    for state in range(world.n_states):
        maxent_reward.append((world.state_index_to_point(state), reward_maxent[state]))
        print("index:", state, " coord:", world.state_index_to_point(state), " reward:", reward[state])

    img_utils.heatmap3d(initial_reward, 'Reward Map - MaxEnt')

    #
    # # maximum casal entropy reinforcement learning (non-causal)
    reward_maxcausal = maxent_causal(world, terminals, trajectories)
    V_maxcausal = S.value_iteration(world.p_transition, reward_maxcausal, discount=0.9, eps=1e-3)
    maxcasual_reward = list(tuple())
    for state in range(world.n_states):
        maxcasual_reward.append((world.state_index_to_point(state), reward_maxent[state]))
        print("index:", state, " coord:", world.state_index_to_point(state), " reward:", reward[state])

    img_utils.heatmap3d(initial_reward, 'Reward Map - MaxCasual')

    print("EXPERT POLICY")
    exp_policy = S.optimal_policy(world, reward, 0.8)
    print(exp_policy)
    print("MAX-ENT-POLICY")
    maxent_policy = S.optimal_policy(world, reward_maxent, 0.8)
    print(maxent_policy)
    print("MAX-CASUALPOLICY")
    maxcasual_policy = S.optimal_policy(world, reward_maxcausal, 0.8)
    print(maxcasual_policy)
    print(reward_maxcausal)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax.title.set_text('Original Reward')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # p = P.plot_state_values(ax, world, reward, **style)
    # P.plot_deterministic_policy(ax, world, S.optimal_policy(world, reward, 0.8), color='red')
    # fig.colorbar(p, cax=cax)
    #
    # ax = fig.add_subplot(122)
    # ax.title.set_text('Recovered Reward')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # p = P.plot_state_values(ax, world, reward_maxent, **style)
    # P.plot_deterministic_policy(ax, world, S.optimal_policy(world, reward_maxent, 0.8), color='red')
    # fig.colorbar(p, cax=cax)
    #
    # fig.tight_layout()
    # plt.show()
    #
    #
    #
    # plt.show()


if __name__ == '__main__':
    main()
