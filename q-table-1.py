import random
from collections import deque

import numpy as np

import gym


def encode_action(action):
    """
    >>> encode_action((0, 0, 0))
    0
    >>> encode_action((1, 0, 0))
    1
    >>> encode_action((1, 1, 0))
    3
    >>> encode_action((1, 1, 1))
    7
    >>> encode_action((1, 1, 4))
    19
    >>> encode_action((1, 0, 4))
    17
    """
    first, second, third = action
    return first + 2 * second + 4 * third


def decode_action(action):
    """
    >>> decode_action(0)
    (0, 0, 0)
    >>> decode_action(1)
    (1, 0, 0)
    >>> decode_action(2)
    (0, 1, 0)
    >>> decode_action(4)
    (0, 0, 1)
    >>> decode_action(9)
    (1, 0, 2)
    """
    third = action // 4
    second = (action - 4 * third) // 2    
    first = action - 4 * third - 2 * second
    return (first, second, third)


class CopyQTable(object):

    actions_space = range(20)

    def __init__(self, gamma=0.99, alpha=0.7):
        self.gamma = gamma
        self.alpha = alpha
        self.Qtable = np.zeros(
            (6, len(self.actions_space))
            ) # 6 state-rows, 20 action-columns

    def epsilon_greedy_action(self, state, epsilon):
        if random.random() < epsilon:
            action = np.random.choice(self.actions_space)
        else:
            Qstate = self.Qtable[state, :]
            maxQstate = np.max(Qstate)
            possible_actions = [a for a in self.actions_space if Qstate[a] >= maxQstate]
            action = np.random.choice(possible_actions)
        return decode_action(action)

    def train(self, state, action, reward, next_state, done):
        maxNextQ = np.max(self.Qtable[next_state, :])
        encoded_action = encode_action(action)
        currentQ = self.Qtable[state, encoded_action]
        update = reward + done * self.gamma * maxNextQ
        self.Qtable[state, encoded_action] = self.alpha * currentQ + (1. - self.alpha) * update


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    env = gym.make('Copy-v0')
#     env = gym.wrappers.Monitor(env, '/tmp/copy-v0', force=True)

    Qtable = CopyQTable(gamma=0.8)

    epsilon, epsilon_decay = 1.0, 0.99
    target = 30.
    performance = deque(maxlen=100)
    performance.append(0.)

    episode = 0
    while episode < 50000 and np.mean(performance) < 25.:
        episode += 1
        state = env.reset()

        steps, rewards, done = 0, [], False
        while not done:
            steps += 1
            action = Qtable.epsilon_greedy_action(state, epsilon)
            # Execute the action and get feedback
            next_state, reward, done, _ = env.step(action)
            Qtable.train(state, action, reward + 0.5, next_state, done) # use shifted reward to update the Q table
            rewards.append(reward)
            state = next_state
        performance.append(np.sum(rewards))
        if epsilon > 0.001 and np.mean(rewards) > 0 and episode >= 1000:
            epsilon *= epsilon_decay * (1.- np.mean(rewards) / target)
        print("episode {} steps {} rewards {} total {} epsilon {}".format(episode, steps, rewards, np.sum(rewards), epsilon))
    print(Qtable.Qtable)
