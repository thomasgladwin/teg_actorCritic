import numpy as np
import matplotlib.pyplot as plt
import Environment
import teg_actorCritic
import importlib
importlib.reload(teg_actorCritic)
importlib.reload(Environment)

#
# Basic search in 2D GridWorld; the observable feature of the Environment is the state itself, i.e., the location coordinates.
#
# A_effect_vec defines the movement caused by actions.
# MapString defines the world: 0 = default state, 1 = start, 2 = terminal, 3 = pit, 4 = wall.

A_effect_vec = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
MapString = '''
4 0 0 0 0 0 0 0 0 0
0 0 0 0 0 4 2 0 0 0
0 0 0 0 0 4 4 4 4 0
0 0 0 0 0 0 0 0 0 0
3 3 3 3 0 3 3 0 0 0
0 1 0 0 0 0 0 0 0 0
'''
random_pit_prob = 0.0
pit_punishment = -2
backtrack_punishment = 0
off_grid_punishment = -1
terminal_reward = 0
observable_features = (True, False, False)

environment = Environment.Environment(A_effect_vec, observable_features, MapString=MapString, random_pit_prob=random_pit_prob)
environment.set_rewards(pit_punishment=pit_punishment, backtrack_punishment=backtrack_punishment, off_grid_punishment=off_grid_punishment, terminal_reward=terminal_reward)

agent = teg_actorCritic.Agent(environment.nFeatures, environment.nA)
agent.critic.lamba0 = 0.5
agent.actor.lamba0 = 0.5

max_episode_length = 1e6
sim = teg_actorCritic.Simulation(max_episode_length)

agent = sim.train(1e3, environment, agent)
route = sim.test(environment, agent)
sim.plots(environment, agent, route)

#
# Random environment per episode (i.e., starting, terminal, and pit locations), relative environment features to learn
#
nR = 6; nC = 10
rStart = np.nan; cStart = np.nan;
rTerm = np.nan; cTerm = np.nan
A_effect_vec = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
random_pit_prob = 0.2
pit_punishment = -1
backtrack_punishment = 0
off_grid_punishment = -1
terminal_reward = 0
observable_features = (False, True, True)

environment = Environment.Environment(A_effect_vec, observable_features, nR=nR, nC=nR, rStart=rStart, cStart=cStart, rTerm=rTerm, cTerm=cTerm, random_pit_prob=random_pit_prob)
environment.set_rewards(pit_punishment=pit_punishment, backtrack_punishment=backtrack_punishment, off_grid_punishment=off_grid_punishment, terminal_reward=terminal_reward)

agent = teg_actorCritic.Agent(environment.nFeatures, environment.nA)
agent.critic.lamba0 = 0
agent.actor.lamba0 = 0

max_episode_length = 1e6
sim = teg_actorCritic.Simulation(max_episode_length)

agent = sim.train(1e3, environment, agent)
route = sim.test(environment, agent)
sim.plots(environment, agent, route)
