import tensorflow as tf
import numpy as np
import random

class Quadcop_Policy:
    def __init__(self, task, learning_rate=0.01, state_size=6,
                 action_size=4, hidden_size=10,
                 name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            #self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            #one_hot_actions = tf.one_hot(self.actions_, action_size)
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name='actions')

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size,
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            #self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            self.task = task
            self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, action):
        next_state, reward, done = self.task.step(action)
        return next_state, reward, done

    def action_sample(self):
        return np.random.uniform(self.task.action_low, self.task.action_high, 4)


from collections import deque


class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


