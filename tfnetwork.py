#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import random
from collections import deque

import numpy as np

import gym
import tensorflow as tf

from utils import encode_action, decode_action


np.random.seed(42)


env = gym.make("Copy-v0")
gamma = 0.8
epsilon = 1.0
num_episodes = 10000
performance = deque(maxlen=100)
performance.append(0.0)


def get_state(state, n_states):
    return np.identity(n_states)[state: state + 1]


def Qaction_predict(X, n_output):
    n_input = int(X.shape[1])
    with tf.variable_scope("predict"):
        weights = tf.Variable(
            tf.random_normal(
                [n_input, n_output], stddev=0.005),
            name="weights")
        Qaction = tf.matmul(X, weights) # shape [env.observation_space.n, env.action_space.n]
    return Qaction


def greedy_action(Qaction):
    return tf.argmax(Qaction, axis=1)


def Qaction_train(Qaction, nextQ):
    loss = tf.reduce_sum(tf.square(nextQ - Qaction))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)
    return updateModel


with tf.Session() as sess:
    # establish the feed-forward part of the network used to choose actions
    X = tf.placeholder(shape=[None, env.observation_space.n], dtype=tf.float32)
    Qout = Qaction_predict(X, n_output=20)
    predict = greedy_action(Qout)
    nextQ = tf.placeholder(shape=[None, 20], dtype=tf.float32)
    updateModel = Qaction_train(Qout, nextQ)

    init = tf.global_variables_initializer()
    sess.run(init)

    episode = 0
    while episode < num_episodes and np.mean(performance) < 25:
        episode += 1
        state = env.reset()

        done = False
        steps = 0
        rewards = []
        # the Q-network
        while not done:
            steps += 1
            # make greedy action and Qaction value predictions
            actions, allQ = sess.run(
                [predict, Qout], feed_dict={
                    X: get_state(state, env.observation_space.n)
                    }
                )
            # decode actions
            actions = [decode_action(a) for a in actions]
            # epsilon-greedy action
            if np.random.rand(1) < epsilon:
                actions[0] = env.action_space.sample()
            # get the new state
            next_state, reward, done, _ = env.step(actions[0])
            # obtain the next Q values by feeding the state through the network
            predNextQ = sess.run(
                Qout, feed_dict={
                    X: get_state(next_state, env.observation_space.n)
                    }
                )
            targetQ = allQ
            targetQ[0, encode_action(actions[0])] = reward + 0.5 * done * gamma * np.max(predNextQ)
            # train the networkd using the target and predicted Q values
            sess.run(
                updateModel, feed_dict={
                    X: get_state(state, env.observation_space.n),
                    nextQ: targetQ
                    }
                )
            rewards.append(reward)
            state = next_state

            if done and epsilon > 0.01 and np.mean(rewards) > 0 and episode >= 1000:
                # reduce the chance of random action as we train the model
                epsilon = 1. / (episode / 50 + 10)

        performance.append(np.sum(rewards))
        print("episode {} steps {} rewards {} total {} epsilon {}".format(episode, steps, rewards, np.sum(rewards), np.around(epsilon, 4)))
