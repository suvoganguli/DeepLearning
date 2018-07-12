import numpy as np
import agents.agent as agent
from task import Task
import matplotlib.pyplot as plt
import sys


# Task: take-off and hover
init_pose = [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]
init_velocities = [0.0, 0.0, 0.0]
init_angle_velocities = [0.0, 0.0, 0.0]
run_time = 10
target_pos = [0.0, 0.0, 100.0]
num_episodes = 20 #1000
best_score = -np.inf

np.random.seed(1234)

task = Task(init_pose, init_velocities, init_angle_velocities, run_time, target_pos)
ddpg = agent.DDPG(task)

reward_all = np.array([], dtype=float)

for i_episode in range(1, num_episodes+1):

    state = ddpg.reset_episode() # start a new episode

    count = 0
    total_reward = 0.0

    while True:

        action = ddpg.act(state)
        next_state, reward, done = task.step(action)
        ddpg.step(action, reward, next_state, done)
        state = next_state

        if count == 0:
            action_save = action
            state_save = state
        else:
            action_save = np.vstack([action_save, action])
            state_save = np.vstack([state_save, state])

        total_reward += reward
        count += 1

        if done:

            nm = action_save.shape
            n = nm[0]
            t = np.arange(0, n) * 0.02

            plt.figure(1)

            plt.subplot(211)
            plt.plot(t, action_save)
            plt.grid('on')
            plt.ylabel('rpm')

            plt.subplot(212)
            plt.plot(t, state_save[:, 2])
            plt.grid('on')
            plt.ylabel('h [m]')
            plt.xlabel('t [sec]')

            plt.pause(0.2)

            if i_episode < num_episodes - 1:
                plt.clf()

            plt.figure(2)
            plt.plot(i_episode, reward, 'bo')
            plt.grid('on')
            plt.xlabel('Iteration')
            plt.ylabel('Episode Reward')

            score = total_reward / float(count) if count else 0.0
            if score > best_score:
                best_score = score

            print("\nEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}) | Rotorspeeds (rpm): {:d} {:d} {:d} {:d} | h (m): {:.1f} ".format(
                i_episode, score, best_score, int(action[0]), int(action[1]), int(action[2]), int(action[3]), state[2] ), end="")

            reward_all = np.concatenate([reward_all, [reward]])

            break

    #sys.stdout.flush()

avg_reward = reward_all/num_episodes

plt.figure(3)
plt.plot(avg_reward)
#plt.ylim(-1000, 1000)
plt.grid('on')
plt.ylabel('Average Reward')
plt.xlabel('Episode Number')

plt.show('all')