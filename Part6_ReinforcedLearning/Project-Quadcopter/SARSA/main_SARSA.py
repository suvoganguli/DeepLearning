from task import Task
import numpy as np
import matplotlib.pyplot as plt

if False:

    class Exp_Agent():
        def __init__(self, task):
            self.task = task

        def act(self):
            new_thrust = 404
            return [new_thrust for x in range(4)]


    runtime = 5.  # time limit of the episode
    init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
    init_velocities = np.array([0., 0., 0.])  # initial velocities
    init_angle_velocities = np.array([0., 0., 0.])  # initial angle velocities

    # Setup
    task = Task(init_pose, init_velocities, init_angle_velocities, runtime)
    agent = Exp_Agent(task)
    done = False
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
    results = {x: [] for x in labels}

    for k in range(200):  # 200 / 50 * 3 = 12 sec (dt = 0.02 sec, action_repeat = 3)
        rotor_speeds = agent.act()
        _, _, done = task.step(rotor_speeds)
        to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
        for ii in range(len(labels)):
            results[labels[ii]].append(to_write[ii])

    plt.figure(1)
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.legend()
    plt.show()

# Linear policy with random perturbation for each policy
if False:
    import sys
    import pandas as pd
    from agents.policy_search import PolicySearch_Agent
    from task import Task

    num_episodes = 200
    target_pos = np.array([0., 0., 10.])
    task = Task(target_pos=target_pos)
    agent = PolicySearch_Agent(task)

    for i_episode in range(1, num_episodes + 1):
        state = agent.reset_episode()  # start a new episode
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(reward, done)
            state = next_state
            if done:
                print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                    i_episode, agent.score, agent.best_score, agent.noise_scale), end="")  # [debug]
                break
        sys.stdout.flush()



# ----------------------------------------------------------
# Project Solution
# ----------------------------------------------------------


if True:

    from task import Task
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from agents.agent import Quadcop_Policy

    target_pos = np.array([0., 0., 10.])
    task = Task(target_pos=target_pos)
    agent = Quadcop_Policy(task)

    # Initialize the simulation
    agent.reset_episode()

    # Memory
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

    # ------------------------------------
    # Hyperparameters
    train_episodes = 100  # max number of episodes to learn from
    max_steps = 200  # max steps in an episode
    gamma = 0.99  # future reward discount

    # Exploration parameters
    explore_start = 1.0  # exploration probability at start
    explore_stop = 0.01  # minimum exploration probability
    decay_rate = 0.0001  # exponential decay rate for exploration prob

    # Network parameters
    hidden_size = 64  # number of units in each Q-network hidden layer
    learning_rate = 0.0001  # Q-network learning rate

    # Memory parameters
    memory_size = 10000  # memory capacity
    batch_size = 20  # experience mini-batch size
    pretrain_length = batch_size  # number experiences to pretrain the memory

    # ------------------------------------

    tf.reset_default_graph()
    mainQN = Quadcop_Policy(name='main', task=task, hidden_size=hidden_size, learning_rate=learning_rate)

    # Take one random step to get the quadcop moving
    action = agent.action_sample()
    state, reward, done = agent.task.step(action)

    memory = Memory(max_size=memory_size)

    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length):

        # Make a random action
        action = agent.action_sample()
        next_state, reward, done = agent.task.step(action)

        if done:
            # The simulation fails so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            memory.add((state, action, reward, next_state))

            # Start new episode
            agent.reset_episode()
            # Take one random step to get the pole and cart moving
            state, reward, done = env.step(agent.action_sample())
        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state

    # ------------------------

    # Now train with experiences

    saver = tf.train.Saver()
    rewards_list = []
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        step = 0
        for ep in range(1, train_episodes):
            total_reward = 0
            t = 0
            while t < max_steps:
                step += 1

                # Explore or Exploit
                explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * step)
                if explore_p > np.random.rand():
                    # Make a random action
                    action = agent.action_sample()
                else:
                    # Get action from Q-network
                    feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)

                # Take action, get new state and reward
                next_state, reward, done = agent.task.step(action)

                total_reward += reward

                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    t = max_steps

                    print('Episode: {}'.format(ep),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_p))
                    rewards_list.append((ep, total_reward))

                    # Add experience to memory
                    memory.add((state, action, reward, next_state))

                    # Start new episode
                    agent.reset_episode()

                    # Take one random step to get the pole and cart moving
                    state, reward, done = agent.step(agent.action_sample())

                else:
                    # Add experience to memory
                    memory.add((state, action, reward, next_state))
                    state = next_state
                    t += 1

                # Sample mini-batch from memory
                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])

                # Train network
                target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})

                # Set target_Qs to 0 for states where episode ends
                end_state = np.zeros(states[0].shape)
                end_state[2] = target_pos[2]
                episode_ends = (next_states == end_state).all(axis=1)
                target_Qs[episode_ends] = (0, 0, 0, 0)

                targets = rewards + gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                   feed_dict={mainQN.inputs_: states,
                                              mainQN.targetQs_: targets,
                                              mainQN.actions_: actions})

    print('done!')
