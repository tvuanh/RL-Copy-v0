import random
from collections import deque

import numpy as np

import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import RandomNormal

from utils import encode_action, decode_action


class Controller(object):

    def __init__(self, n_input, n_output, gamma=1., batch_size=25, model_instances=10):
        self.n_input = n_input
        self.n_output = n_output
        self.action_space = range(n_output)
        self.gamma = gamma
        self.batch_size = batch_size
        self.model_instances = model_instances
        self.memory = deque(maxlen=100 * batch_size * model_instances)

        # action neural network
        self.action_models = []
        for _ in range(self.model_instances):
            model = Sequential()
            model.add(
                Dense(
                    self.n_output, input_dim=self.n_input, activation="linear",
                    kernel_initializer=RandomNormal(
                        mean=0.0, stddev=0.005, seed=None
                        ),
                    use_bias=False
                    )
                )
            model.compile(
                loss="mse", optimizer=Adam(lr=0.01, decay=0.01)
                )
            self.action_models.append(model)

    def preprocess_state(self, state):
        return np.identity(self.n_input)[state : state + 1]

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append(
            (
                self.preprocess_state(state),
                encode_action(action),
                reward,
                self.preprocess_state(next_state),
                done
                )
            )

    def optimal_action(self, state):
        if len(self.memory) < self.model_instances * self.batch_size:
            action = np.random.choice(self.action_space)
        else:
            s = self.preprocess_state(state)
            Qs = [model.predict(s)[0] for model in self.action_models]
            # estimate means and sigmas then draw random events to smoothen the sampling
            means, sigmas = np.mean(Qs, axis=0), np.sqrt(np.var(Qs, axis=0, ddof=1))
            draws = means + sigmas * np.random.randn(len(means))
            action = np.argmax(draws)
        return decode_action(action)

    def replay(self):
        for model in self.action_models:
            if len(self.memory) <= self.model_instances * self.batch_size:
                continue
            else:
                minibatch = random.sample(self.memory, k=self.batch_size)
            x_batch, y_batch = list(), list()
            for state, action, reward, next_state, _ in minibatch:
                y_target = model.predict(state)
                y_target[0, action] = reward + self.gamma * np.max(
                    model.predict(next_state)
                    )
                x_batch.append(state[0])
                y_batch.append(y_target[0])
            model.fit(
                np.array(x_batch), np.array(y_batch), batch_size=len(x_batch),
                verbose=False)


def play(episodes):
    env = gym.make("Copy-v0")
    controller = Controller(
        n_input=env.observation_space.n, n_output=20)

    benchmark = 25
    scores = deque(maxlen=100)
    scores.append(0.0)

    episode = 0
    while episode < episodes and np.mean(scores) < benchmark:
        episode += 1
        state = env.reset()
        done = False
        rewards, steps = [], 0
        while not done:
            action = controller.optimal_action(state)
            next_state, reward, done, _ = env.step(action)
            controller.memorize(state, action, reward, next_state, done)
            state = next_state
            rewards.append(reward)
            steps += 1
        scores.append(np.sum(rewards))

        print(
            "episode {} steps {} rewards {} average score {}".format(
                episode, steps, rewards, np.mean(scores)
                )
            )
        controller.replay()
    return episode


if __name__ == "__main__":
    episodes = 10000
    nplays = 1
    results = np.array([play(episodes) for _ in range(nplays)])
    success = results < episodes
    print("Total number of successful plays is {}/{}".format(np.sum(success), nplays))
    print("Average number of episodes before success per play {}".format(np.mean(results[success])))
