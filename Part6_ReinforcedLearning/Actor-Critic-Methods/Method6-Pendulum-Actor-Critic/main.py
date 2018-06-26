import tensorflow as tf
import numpy as np
import gym
import agents.agent as agent

# ===========================
#   Agent Training
# ===========================


def train(sess, env, ddpg):

    summary_dir = './results'
    max_episodes = 500
    max_episode_len = 1000

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    #writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Initialize replay memory
    buffer_size = ddpg.buffer_size
    batch_size = ddpg.batch_size
    replay_buffer = agent.ReplayBuffer(buffer_size, batch_size)

    for i in range(max_episodes):

        # Reset
        state = ddpg.reset_episode()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(max_episode_len):

            env.render()

            # Get action from state
            action = ddpg.act(state)

            # Take a step (Get state, action, reward, next_state, done)
            next_state, reward, done, info = env.step(action)

            # Keep adding experience until buffer full
            replay_buffer.add(state, action, reward, next_state, done)

            # If buffer full:
            #   - Get batches of experience (states, actions, reward, next_state, done)
            #   - Learn

            if replay_buffer.__len__() >  ddpg.batch_size:

                experiences = replay_buffer.sample(batch_size)

                ddpg.learn(experiences)

                ep_ave_max_q += ddpg.Qmax   # IS THIS RIGHT Q TO LOOK AT?

            state = next_state
            ep_reward += reward

            if done:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / (float(j+1))
                })

                #writer.add_summary(summary_str, i)
                #writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Eposide Length: {:d}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j+1)), j+1 ))
                break


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


def main():
    with tf.Session() as sess:

        env = gym.make('Pendulum-v0')

        random_seed = 1234
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        env.seed(random_seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        action_low = -action_bound
        action_high = action_bound

        ddpg = agent.DDPG(env, state_dim, action_dim, action_low, action_high)

        train(sess, env, ddpg)


if __name__ == '__main__':

    main()