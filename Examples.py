import numpy as np
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
agent.critic.set_lambda(0.5)
agent.actor.set_lambda(0.5)

max_episode_length = 1e4
sim = teg_actorCritic.Simulation(max_episode_length)

agent = sim.train(1e3, environment, agent)
routes = sim.test(environment, agent, 5)
sim.plots(environment, agent, routes)

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
agent.critic.set_lambda(0)
agent.actor.set_lambda(0)

max_episode_length = 1e6
sim = teg_actorCritic.Simulation(max_episode_length)

agent = sim.train(1e3, environment, agent)
routes = sim.test(environment, agent, 5)
sim.plots(environment, agent, routes)

#
# The Line.
#
# A_effect_vec defines the movement caused by actions.
# MapString defines the world: 0 = default state, 1 = start, 2 = terminal, 3 = pit, 4 = wall.

# A_effect_vec = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
MapString = '''
4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
4 2 3 3 3 3 3 3 1 0 0 0 0 0 4
4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
'''
nItsPerCoeffComb = 30
routes_per_c = []
for lambda_it_c in [0.01, 0.99]:
    for lambda_it_a in [lambda_it_c]:
        ep_lens = []
        for iIt in range(nItsPerCoeffComb):
            A_effect_vec = [[-1, 0], [0, -1], [0, 1], [1, 0]]
            random_pit_prob = 0.0
            pit_punishment = -1
            backtrack_punishment = 0
            off_grid_punishment = -1
            terminal_reward = 1
            observable_features = (True, False, False)
            nEpisodes = 5e2
            max_episode_length = 1e4
            lambda_critic = lambda_it_c
            lambda_actor = lambda_it_a

            environment = Environment.Environment(A_effect_vec, observable_features, MapString=MapString, random_pit_prob=random_pit_prob)
            environment.set_rewards(pit_punishment=pit_punishment, backtrack_punishment=backtrack_punishment, off_grid_punishment=off_grid_punishment, terminal_reward=terminal_reward)

            agent = teg_actorCritic.Agent(environment.nFeatures, environment.nA)
            agent.critic.set_lambda(lambda_critic)
            agent.actor.set_lambda(lambda_actor)

            sim = teg_actorCritic.Simulation(max_episode_length)

            agent = sim.train(nEpisodes, environment, agent)
            routes = sim.test(environment, agent, 5)
            #sim.plots(environment, agent, routes)

            routes_per_c.append(routes.copy())
            ep_lens.append(sim.ep_len.copy())

        ep_len_smoothed = np.array(ep_lens).mean(axis=0)
        sim.ep_len = ep_len_smoothed
        sim.plots(environment, agent, routes)
