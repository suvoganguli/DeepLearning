import numpy as np
from task import Task
from collections import defaultdict, deque
import sys

class Quadcop_Policy():
    def __init__(self, task):

        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Episode variables
        self.reset_episode()


    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, action):
        next_state, reward, done = self.task.step(action)
        return next_state, reward, done


    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        return action


    def update_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))


    def epsilon_greedy_probs(self, Q_s, i_episode, eps=None):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps

        nA = self.action_size
        policy_s = np.ones(nA) * epsilon / nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / nA)
        return policy_s


    def sarsa(self, num_episodes, alpha, gamma=1.0):

        # initialize action-value function (empty dictionary of arrays)

        nA = self.action_size

        Q = defaultdict(lambda: np.zeros(nA))

        # initialize performance monitor
        plot_every = 100
        tmp_scores = deque(maxlen=plot_every)
        scores = deque(maxlen=num_episodes)

        # loop over episodes
        for i_episode in range(1, num_episodes + 1):
            # monitor progress
            if i_episode % 100 == 0:
                print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
                sys.stdout.flush()

            # initialize score
            score = 0

            # begin an episode, observe S
            state = self.reset_episode()

            nS = self.state_size

            # get epsilon-greedy action probabilities
            policy_s = self.epsilon_greedy_probs(Q[nS], i_episode)

            # pick action
            #action = np.random.choice(np.arange(nA), p=policy_s)
            action_all = np.random.uniform(self.action_low, self.action_high) * policy_s
            action = np.round(action_all[0])
            action = [action, action, action, action]

            # --------------------------------------------------------
            # Getting stuck here. "action" is of size 1 (between 0 and 3)
            # But what I need is a 4x1 vector ranging between
            # task.action_low and task.action_high to run:
            # "next_state, reward, done = self.task.step(action)"
            # --------------------------------------------------------

            # limit number of time steps per episode
            for t_step in np.arange(300):

                # take action A, observe R, S'
                next_state, reward, done = self.task.step(action)
                #next_state, reward, done = self.step(action)

                # add reward to score
                score += reward

                if not done:

                    # Convert to tuple to avoid errors
                    state = tuple(state)
                    next_state = tuple(next_state)

                    # get epsilon-greedy action probabilities
                    policy_s = self.epsilon_greedy_probs(Q[next_state], i_episode)

                    # pick next action A'
                    next_action = np.random.uniform(self.action_low, self.action_high) * policy_s

                    # Convert to tuple to avoid errors
                    action = tuple(action)
                    next_action = tuple(next_action)

                    # update TD estimate of Q
                    Q[state][action] = self.update_Q(Q[state][action], Q[next_state][next_action],
                                                reward, alpha, gamma)

                    # S <- S'
                    state = next_state

                    # A <- A'
                    action = next_action

                if done:
                    # update TD estimate of Q
                    Q[state][action] = self.update_Q(Q[state][action], 0, reward, alpha, gamma)

                    # append score
                    tmp_scores.append(score)
                    break

            if (i_episode % plot_every == 0):
                scores.append(np.mean(tmp_scores))
        # plot performance
        #plt.plot(np.linspace(0, num_episodes, len(scores), endpoint=False), np.asarray(scores))
        #plt.xlabel('Episode Number')
        #plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
        #plt.show()
        # print best 100-episode performance
        print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
        return Q