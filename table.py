#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import random
from collections import deque

import numpy as np

import gym

from utils import encode_action, decode_action


class CopyQTable(object):

    actions_space = range(20)

    def __init__(self, gamma=1.):
        self.gamma = gamma
        self.Qmean = np.zeros(
            (6, len(self.actions_space))
            ) # 6 state-rows, 20 action-columns
        self.counters = np.ones(self.Qmean.shape)
        self.Sr2 = np.zeros(self.Qmean.shape) # sum of squared rewards
        self.Qvar = np.ones(self.Qmean.shape) * np.inf

    def epsilon_greedy_action(self, state, epsilon):
        if random.random() < epsilon:
            action = np.random.choice(self.actions_space)
        else:
            Qstate = self.Qmean[state, :]
            maxQstate = np.max(Qstate)
            possible_actions = [a for a in self.actions_space if Qstate[a] >= maxQstate]
            action = np.random.choice(possible_actions)
        return decode_action(action)

    def optimal_action(self, state):
        counts = self.counters[state, :]
        if np.min(counts) < 10:
            action = np.random.choice(self.actions_space)
        else:
            means, variances = self.Qmean[state, :], self.Qvar[state, :]
            upper_bounds = means + 2. * np.sqrt(np.divide(variances, counts))
            maxQstate = np.max(upper_bounds)
            possible_actions = [a for a in self.actions_space if upper_bounds[a] >= maxQstate]
            action = np.random.choice(possible_actions)
        return decode_action(action)

    def train(self, state, action, reward, next_state):
        maxNextQ = np.max(self.Qmean[next_state, :])
        encoded_action = encode_action(action)
        currentQ = self.Qmean[state, action]
        update = reward + self.gamma * maxNextQ
        self.update_mean(state, encoded_action, update)
        self.update_sum_squared_rewards(state, encoded_action, update)
        self.update_variance(state, encoded_action, update)
        self.counters[state, encoded_action] += 1

    def update_mean(self, state, action, update):
        counts = self.counters[state, action]
        Qm = self.Qmean[state, action]
        self.Qmean[state, action] += (update - Qm) / counts

    def update_sum_squared_rewards(self, state, action, update):
        self.Sr2[state, action] += update * update

    def update_variance(self, state, action, update):
        counts = self.counters[state, action]
        if counts > 1:
            sr2 = self.Sr2[state, action]
            Qm = self.Qmean[state, action]
            self.Qvar[state, action] = (sr2 - counts * Qm * Qm) / (counts - 1)


if __name__ == '__main__':
    env = gym.make('Copy-v0')
#     env = gym.wrappers.Monitor(env, '/tmp/copy-v0', force=True)

    Qtable = CopyQTable(gamma=0.8)

    epsilon = 1.0
    target = 30.
    performance = deque(maxlen=100)
    performance.append(0.)

    episode = 0
    while episode < 10000 and np.mean(performance) < 25.:
        episode += 1
        state = env.reset()

        steps, rewards, done = 0, [], False
        while not done:
            steps += 1
            # action = Qtable.epsilon_greedy_action(state, epsilon)
            action = Qtable.optimal_action(state)
            # Execute the action and get feedback
            next_state, reward, done, _ = env.step(action)
            Qtable.train(state, action, reward + 0.5, next_state) # use shifted reward to update the Q table
            rewards.append(reward)
            state = next_state
        performance.append(np.sum(rewards))
        if epsilon > 0.001 and np.mean(rewards) > 0 and episode >= 1000:
            epsilon *= 1.- np.mean(rewards) / target
        print("episode {} steps {} rewards {} total {} epsilon {}".format(episode, steps, rewards, np.sum(rewards), epsilon))
    print(Qtable.Qmean)
