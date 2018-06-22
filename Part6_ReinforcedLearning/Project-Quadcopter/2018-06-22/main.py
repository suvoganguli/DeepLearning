import tensorflow as tf
import numpy as np
import agents.agent as agent
from task import Task

# ===========================
#   Agent Training
# ===========================

def train(sess, ddpg):

    summary_dir = './results'
    max_episodes = 100


    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Initialize replay memory
    buffer_size = ddpg.buffer_size
    batch_size = ddpg.batch_size
    replay_buffer = agent.ReplayBuffer(buffer_size, batch_size)

    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    for i in range(max_episodes):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j))))
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

        init_pose = [0.0, 0.0, 0.0]
        init_velocities = [0.0, 0.0, 0.0]
        init_angle_velocities = [0.0, 0.0, 0.0]
        run_time = 5
        target_pos = [0.0, 0.0, 10.0]

        ddpg = agent.DDPG(Task(init_pose, init_velocities, init_angle_velocities, run_time, target_pos))

        train(sess, ddpg)


if __name__ == '__main__':

    # args = {'actor-lr': 0.0001,
    #         'critic-lr': 0.001,
    #         'gamma': 0.99,
    #         'tau': 0.001,
    #         'buffer-size': 1000000,
    #         'minibatch-size': 64,
    #         'random-seed': 1234,
    #         'max-episodes': 10,
    #         'max-episode-len': 100,
    #         'monitor-dir': './results',
    #         'summary-dir': './results'}

    main()