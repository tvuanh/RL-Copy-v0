from collections import deque

import numpy as np

import gym

from lisa import rl


def play(episodes=10000):
    env = gym.make('Copy-v0')

    states = range(6)
    actions = tuple(
        [(i, j, k) for i in (0, 1) for j in (0, 1) for k in range(5)]
    )
    Qtable = rl.QTable(states=states, actions=actions, gamma=0.8)

    performance = deque(maxlen=100)
    performance.append(0.)

    episode = 0
    while episode < episodes and np.mean(performance) < 25.:
        episode += 1
        state = env.reset()

        steps, rewards, done = 0, [], False
        while not done:
            steps += 1
            action = Qtable.predict(state)
            next_state, reward, done, _ = env.step(action)
             # use shifted reward to update the Q table
            Qtable.fit(state, action, reward + 0.5, next_state)
            rewards.append(reward)
            state = next_state
        performance.append(np.sum(rewards))
        print("episode {} steps {} rewards {} total {}".format(episode, steps, rewards, np.sum(rewards)))

    return episode


if __name__ == '__main__':

    episodes = 3000
    results = np.array([play(episodes) for _ in range(200)])
    success = results < episodes
    print("Total number of successful plays is {}".format(np.sum(success)))
    print("Average number of episodes before success per play {}".format(np.mean(results[success])))
