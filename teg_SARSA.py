# SARSA(lambda) model with a linear value function on indicator features
# Episodic.
# From Sutton & Barto chapter 13.
import numpy as np
import matplotlib.pyplot as plt
import Environment
import importlib
importlib.reload(Environment)

class SARSA:
    def __init__(self, nFeatures, nA):
        self.w = np.zeros((nFeatures, nA))
        self.z = np.zeros((nFeatures, nA))
        self.nA = nA
        self.alpha0 = 0.1
        self.lambda0 = 0.5
        self.gamma0 = 0.5
        self.action_error_prob = 0.1
    def init_episode(self):
        self.z = 0 * self.z
    def update(self, a, r, feature_vec, feature_vec_new, terminal):
        # Feature vec contains indicators
        q = np.dot(feature_vec[:,0], self.w[:, a])
        if terminal == False:
            q_new = np.dot(feature_vec_new[:,0], self.w[:, a])
        else:
            q_new = 0
        self.delta0 = r + self.gamma0 * q_new - q
        self.z[:, a] = self.gamma0 * self.lambda0 * self.z[:, a] + feature_vec[:,0]
        self.w[:, a] = self.w[:, a] + self.alpha0 * self.delta0 * self.z[:, a]
    def act_on_policy(self, feature_vec, allowed_actions=[], error_free=False):
        if len(allowed_actions) == 0:
            allowed_actions = np.array(range(self.nA))
        allowed_actions = allowed_actions.astype(int)
        action_error_rnd = np.random.rand()
        if action_error_rnd < self.action_error_prob and error_free == False:
            self.a = np.random.choice(allowed_actions)
        else:
            sa_q = np.array([])
            for b in allowed_actions:
                q = np.dot(feature_vec[:,0], self.w[:, b])
                sa_q = np.append(sa_q, q)
            self.a = allowed_actions[np.argmax(sa_q)]
        return self.a

class Simulation:
    def __init__(self, max_episode_length):
        self.ep_len = np.array([])
        self.max_episode_length = max_episode_length
        pass
    def train(self, nEpisodes, environment, sarsa):
        environment.init_episode()
        sarsa.init_episode()
        self.ep_len = np.array([])
        iEpisode = 0
        t_ep = 0
        while iEpisode < nEpisodes:
            print(iEpisode, '. ', end='', sep='')
            print('(', environment.s_r, ', ', environment.s_c, '). ', end='', sep='')
            feature_vec, allowed_actions = environment.state_to_features()
            print(allowed_actions, '. ', sep='', end='')
            a = sarsa.act_on_policy(feature_vec, allowed_actions=allowed_actions)
            r, terminal = environment.respond_to_action(a)
            print('a = ', a, '. r = ', r, '. ', end='\n', sep='')
            feature_vec_new, allowed_actions_new = environment.state_to_features()
            sarsa.update(a, r, feature_vec, feature_vec_new, terminal)
            if t_ep > self.max_episode_length:
                print('XXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXX')
                print('Episode failed.')
                print('XXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXX')
                terminal = True
            if terminal == True:
                environment.init_episode()
                sarsa.init_episode()
                self.ep_len = np.append(self.ep_len, t_ep)
                t_ep = 0
                iEpisode = iEpisode + 1
            t_ep = t_ep + 1
        return sarsa
    def test(self, environment, sarsa):
        environment.init_episode()
        terminal = False
        route = np.array([environment.s_r, environment.s_c])
        t = 0
        while not terminal:
            print(t, ': ', end='')
            feature_vec, allowed_actions = environment.state_to_features()
            a = sarsa.act_on_policy(feature_vec, allowed_actions=allowed_actions, error_free=True)
            print('(', environment.s_r, environment.s_c, ')')
            r, terminal = environment.respond_to_action(a)
            if not terminal:
                route = np.append(route, [environment.s_r, environment.s_c])
            t = t + 1
        route = route.reshape(int(len(route)/2), 2)
        return route
    def plots(self, environment, sarsa, route):
        W = np.max(sarsa.w[:(environment.nR * environment.nC)], axis=1).reshape(environment.nR, environment.nC)
        a = environment.nR * environment.nC
        b = a + 9
        W_local = np.max(sarsa.w[a:b], axis=1).reshape(3, 3)
        a = environment.nR * environment.nC + 9 - 0
        b = a + 6 + 0
        figs, ax = plt.subplots(4, 1)
        ax[0].plot(self.ep_len)
        ax[1].pcolormesh(W)
        ax[2].pcolormesh(environment.pit_map + environment.wall_map * 2)
        if len(route) > 0:
            ax[2].scatter(route[:, 1] + 0.5, route[:, 0] + 0.5)
            ax[2].plot(route[:, 1] + 0.5, route[:, 0] + 0.5)
            ax[2].xaxis.set_ticks(ticks=np.array([range(environment.nC)]).reshape(environment.nC) + 0.5, labels=np.array([range(environment.nC)]).reshape(environment.nC))
            ax[2].yaxis.set_ticks(ticks=np.array([range(environment.nR)]).reshape(environment.nR) + 0.5, labels=np.array([range(environment.nR)]).reshape(environment.nR))
        ax[3].pcolormesh(W_local)
        plt.show()

# Inits
nR = 9; nC = 9;
rStart = 1; cStart = 3;
#rStart = np.nan; cStart = np.nan;
rTerminal = 3; cTerminal = 7
#rTerminal = np.nan; cTerminal = np.nan
#rTerminal = 7; cTerminal = 7
#A_effect_vec = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
A_effect_vec = [[0, 1], [0, -1],[1, 0], [-1, 0]]
wind_vec = np.zeros((nC))
#wind_vec[np.array([3, 4, 5, 6])] = 1
pit_vec = np.array([])
#pit_vec = np.array([[0, 4], [0, 5], [0, 6], [4, 4], [4, 5], [4, 6]])
pit_prob = 0.0
pit_punishment = -1
backtrack_punishment = 0
off_grid_punishment = -1
terminal_reward = 0
wall_vec = np.array([])
# wall_vec = np.array([[4, 4], [5, 4], [6, 4], [7, 4], [8, 4]]) # , [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3]
environment = Environment.Environment(nR, nC, rStart, cStart, rTerminal, cTerminal, A_effect_vec)
environment.define_specifics(wind_vec, pit_vec, pit_prob, wall_vec, pit_punishment, backtrack_punishment, off_grid_punishment, terminal_reward, [True, False, False])

obs_ind = environment.get_observables_indices()

sarsa = SARSA(environment.nFeatures, environment.nA)
sarsa.lambda0 = 0.5

max_episode_length = 1e6
sim = Simulation(max_episode_length)

sarsa = sim.train(1e4, environment, sarsa)
route = sim.test(environment, sarsa)
sim.plots(environment, sarsa, route)

#

environment.rStart = 1
environment.cStart = 3
environment.rTerm0 = 6
environment.cTerm0 = 4
environment.rTerm = environment.rTerm0
environment.cTerm = environment.cTerm0
environment.pit_vec = np.array([])
environment.pit_vec = np.array([[2, 3], [3, 3]])
environment.pit_prob = 0
route = sim.test(environment, sarsa)
sim.plots(environment, sarsa, route)
