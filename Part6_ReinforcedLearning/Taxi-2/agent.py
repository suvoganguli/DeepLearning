import numpy as np
from collections import defaultdict
import gym
env = gym.make('Taxi-v2')

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.i_episode = 1
        self.env = env
        self.alpha = 0.1

    def epsilon_greedy_probs(self, Q_s):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """

        epsilon = 1/self.i_episode
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s


    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        #return np.random.choice(self.nA)


        Q_s = self.Q[state]
        policy_s = self.epsilon_greedy_probs(Q_s)

        action = np.random.choice(self.nA, p=policy_s)
        return action


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        self.Q[state][action] += self.alpha * (reward + np.max(self.Q[next_state]) - self.Q[state][action] )
        self.i_episode += 1
