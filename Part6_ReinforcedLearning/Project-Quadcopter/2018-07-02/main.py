import tensorflow as tf
import numpy as np
import agents.agent as agent
from task import Task
import matplotlib.pyplot as plt
import random

# ===========================
#   Agent Training
# ===========================

def train(sess, ddpg, max_episodes):

    summary_dir = './results'
    #max_episodes = 2
    max_episode_len = 500

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    #writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Initialize replay memory
    buffer_size = ddpg.buffer_size
    batch_size = ddpg.batch_size
    replay_buffer = agent.ReplayBuffer(buffer_size, batch_size)

    reward_all = np.array([], dtype=float)

    for i in range(max_episodes):

        # Reset
        state = ddpg.reset_episode()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(max_episode_len):

            # Get action from state
            action = ddpg.act(state)

            # Ideal Solution
            #action = np.ones(4)*405

            # Take a step (Get state, action, reward, next_state, done)
            next_state, reward, done = ddpg.task.step(action)

            # Keep adding experience until buffer full
            replay_buffer.add(state, action, reward, next_state, done)

            # If buffer full:
            #   - Get batches of experience (states, actions, reward, next_state, done)
            #   - Learn

            if replay_buffer.__len__() >  ddpg.batch_size:

                experiences = replay_buffer.sample(batch_size)
                ddpg.learn(experiences)
                ep_ave_max_q += ddpg.Qmax   # IS THIS RIGHT Q TO LOOK AT?

            if j == 0:
                action_save = action
                state_save = state
            else:
                action_save = np.vstack([action_save, action])
                state_save = np.vstack([state_save, state])

            state = next_state
            ep_reward += reward

            if done:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / (float(j+1))
                })

                #writer.add_summary(summary_str, i)
                #writer.flush()

                #print(state)
                #print(action)

                print('| Reward: {:d} | Episode: {:d} | Episode Length: {:d} | Rotorspeeds: {:d} {:d} {:d} {:d}'.format(int(ep_reward), \
                        i, j, int(action[0]), int(action[1]), int(action[2]), int(action[3]) ))


                if False:
                    nm = action_save.shape
                    n = nm[0]
                    t = np.arange(0,n)*0.02

                    plt.figure(1)

                    plt.subplot(211)
                    plt.plot(t,action_save)
                    plt.grid('on')
                    plt.ylabel('rpm')

                    plt.subplot(212)
                    plt.plot(t,state_save[:,2])
                    plt.grid('on')
                    plt.ylabel('h [m]')
                    plt.xlabel('t [sec]')

                    plt.pause(0.2)

                    if i == max_episodes-1:
                        plt.figure(1)
                        plt.savefig('./results/fig1.jpg')

                    if i < max_episodes-1:
                        plt.clf()

                    plt.figure(2)
                    plt.plot(i, ep_reward, 'bo')
                    plt.grid('on')
                    plt.xlabel('Iteration')
                    plt.ylabel('Episode Reward')

                    if i == max_episodes-1:

                        plt.figure(2)
                        plt.savefig('./results/fig2.jpg')

                reward_all = np.concatenate([reward_all, [ep_reward]])

                break

    return reward_all


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


def main(max_episodes):


    with tf.Session() as sess:

        # Task: take-off and hover
        init_pose = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        init_velocities = [0.0, 0.0, 0.0]
        init_angle_velocities = [0.0, 0.0, 0.0]
        run_time = 20
        target_pos = [0.0, 0.0, 10.0]

        np.random.seed(1234)
        random.seed(2345)

        ddpg = agent.DDPG(Task(init_pose, init_velocities, init_angle_velocities, run_time, target_pos))

        reward_all = train(sess, ddpg, max_episodes)

        print('done')

        return reward_all


if __name__ == '__main__':

    max_episodes = 100

    reward_all = main(max_episodes)
    avg_reward = reward_all/max_episodes

    plt.figure(3)
    plt.plot(avg_reward)
    plt.ylim(-1000, 1000)
    plt.grid('on')
    plt.ylabel('Average Reward')
    plt.xlabel('Episode Number')
    plt.pause(0.02)

