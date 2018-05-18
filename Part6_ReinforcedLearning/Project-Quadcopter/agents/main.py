from agents.agent import Quadcop_Policy
from task import Task
import numpy as np

num_episodes = 10
target_pos = np.array([0., 0., 10.])
task = Task(target_pos=target_pos)
agent = Quadcop_Policy(task)

# x = np.random.uniform(0,10,2)
# print(x)

alpha = 0.1
Q = agent.sarsa(num_episodes, alpha, gamma=1.0)