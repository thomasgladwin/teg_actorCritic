# SARSA(lambda) model with a linear value function on indicator features
# Episodic.
# From Sutton & Barto chapter 13.
import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, nR, nC, rStart, cStart, rTerm, cTerm, A_effect_vec):
        self.s_r = 0
        self.s_c = 0
        self.nR = nR
        self.nC = nC
        self.A_effect_vec = A_effect_vec
        self.nA = len(self.A_effect_vec)
        self.rStart = rStart
        self.cStart = cStart
        self.rTerm0 = rTerm # For np.nan random-per-episode terminal points
        self.cTerm0 = cTerm
        self.rTerm = rTerm
        self.cTerm = cTerm
        self.mem_length = 9 # Avoid loops (exclude actions or punish? Punish affects mean r)
        self.memory = []
    def define_specifics(self, wind_vec, pit_vec, pit_prob=0.0, wall_vec=np.array([]), pit_punishment=-1, backtrack_punishment=-1, terminal_reward=0, observable_features=[True, False, False]):
        self.wind_vec = wind_vec
        self.pit_vec = pit_vec
        self.pit_map = np.zeros((self.nR, self.nC))
        self.pit_prob = pit_prob
        self.wall_map = np.zeros((self.nR, self.nC))
        for i_wall in range(wall_vec.shape[0]):
            self.wall_map[wall_vec[i_wall][0]][wall_vec[i_wall][1]] = 1
        self.pit_punishment = pit_punishment
        self.backtrack_punishment = backtrack_punishment
        self.terminal_reward = terminal_reward
        self.observable_features = observable_features # Coordinates, local pits, goal direction
        feature_vec, allowed_actions = self.state_to_features()
        self.nFeatures = int(np.prod(feature_vec.shape))
    def init_episode(self):
        self.s_r = self.rStart
        self.s_c = self.cStart
        while True:
            if np.isnan(self.rTerm0):
                self.rTerm = np.random.randint(0, self.nR)
            if np.isnan(self.cTerm0):
                self.cTerm = np.random.randint(0, self.nC)
            if np.isnan(self.rStart):
                self.s_r = np.random.randint(0, self.nR)
            if np.isnan(self.cStart):
                self.s_c = np.random.randint(0, self.nC)
            if self.s_r != self.rTerm or self.s_c != self.cTerm:
                break
        # Make pits: pre-set and random
        self.pit_map = np.zeros((self.nR, self.nC))
        for i_pit in range(pit_vec.shape[0]):
            self.pit_map[self.pit_vec[i_pit][0]][self.pit_vec[i_pit][1]] = 1
        for r in range(self.nR):
            for c in range(self.nC):
                die = np.random.rand()
                if die < self.pit_prob:
                    if not (np.abs(r - self.rTerm) <= 1 and np.abs(c - self.cTerm) <= 1):
                        self.pit_map[r, c] = 1
        self.memory = [(self.s_r, self.s_c) for n in range(self.mem_length)]
    def f_into_wall(self, new_s_r, new_s_c):
        in_wall = self.wall_map[new_s_r, new_s_c] == 1
        return in_wall
    def f_pit(self):
        in_pit = self.pit_map[self.s_r, self.s_c] == 1
        return in_pit
    def f_terminal(self):
        if self.s_r == self.rTerm and self.s_c == self.cTerm:
            return True
        else:
            return False
    def f_backtracking(self):
        if (self.s_r, self.s_c) in self.memory:
            return True
        else:
            return False
    def state_to_gridcoords(self):
        X = np.zeros((self.nR, self.nC))
        X[self.s_r, self.s_c] = 1
        X = X.reshape(self.nR * self.nC).copy()
        return X
    def state_to_local(self):
        X = np.zeros((3, 3))
        for idr in range(3):
            dr = -1 + idr
            for idc in range(3):
                dc = -1 + idc
                r = self.s_r + dr
                c = self.s_c + dc
                if r >= 0 and r < self.nR and c >= 0 and c < self.nC:
                    if self.pit_map[r, c] == 1:
                        X[idr, idc] = 1
        X = X.reshape(np.prod(X.shape)).copy()
        return X
    def state_to_goalrelative(self):
        if False:
            X = np.zeros((2, 3))
            if self.s_r < self.rTerm:
                X[0,0] = 1
            if self.s_r == self.rTerm:
                X[0,1] = 1
            if self.s_r > self.rTerm:
                X[0,2] = 1
            if self.s_c < self.cTerm:
                X[1,0] = 1
            if self.s_c == self.cTerm:
                X[1,1] = 1
            if self.s_c > self.cTerm:
                X[1,2] = 1
        else:
            X = []
            for r in range(-1, 2):
                for c in range(-1, 2):
                    dr = 0
                    if self.s_r - self.rTerm > 0:
                        dr = 1
                    else:
                        dr = -1
                    dc = 0
                    if self.s_c - self.cTerm > 0:
                        dc = 1
                    else:
                        dc = -1
                    if r == dr and c == dc:
                        X.append(1)
                    else:
                        X.append(0)
            X = np.array(X)
        X = X.reshape(np.prod(X.shape)).copy()
        return X
    def get_observables_indices(self):
        a = 0
        b = self.nR * self.nC
        c = b
        d = b + 9
        e = d
        f = e + 6
        obs_ind = [(a, b), (c, d), (e, f)]
        return obs_ind
    def state_to_features(self):
        X_gridCoordinates = self.state_to_gridcoords()
        if self.observable_features[0] == False:
            X_gridCoordinates = 0 * X_gridCoordinates
        X_local = self.state_to_local()
        if self.observable_features[1] == False:
            X_local = 0 * X_local
        X_goal = self.state_to_goalrelative()
        if self.observable_features[2] == False:
            X_goal = 0 * X_goal
        # Create full feature vector (column)
        X = np.append(X_gridCoordinates, X_local)
        X = np.append(X, X_goal)
        X = X.reshape((len(X)))
        allowed_actions = np.array([])
        for a in range(len(self.A_effect_vec)):
            new_s_r = self.s_r + self.A_effect_vec[a][0]
            new_s_c = self.s_c + self.A_effect_vec[a][1]
            if (new_s_r >= 0 and new_s_r < self.nR) and (new_s_c >= 0 and new_s_c < self.nC) and not self.f_into_wall(new_s_r, new_s_c):
                if self.backtrack_punishment < 0 and (not (new_s_r, new_s_c) in self.memory):
                    allowed_actions = np.append(allowed_actions, a)
        return X, allowed_actions
    def respond_to_action(self, a):
        new_s_r = self.s_r + self.A_effect_vec[a][0]
        new_s_c = self.s_c + self.A_effect_vec[a][1]
        new_s_c = np.min([self.nC - 1, np.max([0, new_s_c])])
        new_s_r = int(new_s_r + self.wind_vec[new_s_c])
        s_r_mem = self.s_r
        s_c_mem = self.s_c
        self.s_r = np.min([self.nR - 1, np.max([0, new_s_r])])
        self.s_c = np.min([self.nC - 1, np.max([0, new_s_c])])
        r = -1
        terminal = False
        if self.f_terminal():
            r = self.terminal_reward
            terminal = True
        else:
            if self.f_pit():
                # terminal = True # Does entering a pit end the episode?
                r = r + self.pit_punishment
            if self.f_backtracking():
                r = r + self.backtrack_punishment
        self.memory.pop(0)
        self.memory.append((self.s_r, self.s_c))
        return r, terminal

class SARSA:
    def __init__(self, nFeatures, nA):
        self.w = np.zeros((nFeatures, nA))
        self.z = np.zeros((nFeatures, nA))
        self.nA = nA
        self.alpha0 = 0.1
        self.lambda0 = 0.5
        self.gamma0 = 0.5
        self.action_error_prob = 0
    def init_episode(self):
        self.z = 0 * self.z
    def update(self, a, r, feature_vec, feature_vec_new, terminal):
        # Feature vec contains indicators
        q = np.dot(feature_vec, self.w[:, a])
        if terminal == False:
            q_new = np.dot(feature_vec_new, self.w[:, a])
        else:
            q_new = 0
        self.delta0 = r + self.gamma0 * q_new - q
        self.z[:, a] = self.gamma0 * self.lambda0 * self.z[:, a] + feature_vec
        self.w[:, a] = self.w[:, a] + self.alpha0 * self.delta0 * self.z[:, a]
    def act_on_policy(self, feature_vec, allowed_actions=[], error_free=True):
        if len(allowed_actions) == 0:
            allowed_actions = np.array(range(self.nA))
        allowed_actions = allowed_actions.astype(int)
        action_error_rnd = np.random.rand()
        if action_error_rnd < self.action_error_prob and error_free == False:
            self.a = np.random.choice(allowed_actions)
        else:
            sa_q = np.array([])
            for b in allowed_actions:
                q = np.dot(feature_vec, self.w[:, b])
                sa_q = np.append(sa_q, q)
            self.a = allowed_actions[np.argmax(sa_q)]
        return self.a

class Simulation:
    def __init__(self, max_episode_length):
        self.ep_len = np.array([])
        self.max_episode_length = max_episode_length
        pass
    def train(self, nEpisodes, environment, SARSA):
        environment.init_episode()
        SARSA.init_episode()
        self.ep_len = np.array([])
        iEpisode = 0
        t_ep = 0
        while iEpisode < nEpisodes:
            print(iEpisode, '. ', end='', sep='')
            print('(', environment.s_r, ', ', environment.s_c, '). ', end='', sep='')
            feature_vec, allowed_actions = environment.state_to_features()
            print(allowed_actions, '. ', sep='', end='')
            a = SARSA.act_on_policy(feature_vec, allowed_actions=allowed_actions)
            r, terminal = environment.respond_to_action(a)
            feature_vec_new, allowed_actions_new = environment.state_to_features()
            SARSA.update(a, r, feature_vec, feature_vec_new, terminal)
            print('a = ', a, '. r = ', r, '. ', end='\n', sep='')
            if t_ep > self.max_episode_length:
                print('XXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXX')
                print('Episode failed.')
                print('XXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXX')
                terminal = True
            if terminal == True:
                environment.init_episode()
                SARSA.init_episode()
                self.ep_len = np.append(self.ep_len, t_ep)
                t_ep = 0
                iEpisode = iEpisode + 1
            t_ep = t_ep + 1
        return SARSA
    def test(self, environment, SARSA):
        environment.init_episode()
        terminal = False
        route = np.array([environment.s_r, environment.s_c])
        t = 0
        while not terminal:
            print(t, ': ', end='')
            feature_vec, allowed_actions = environment.state_to_features()
            a = SARSA.act_on_policy(feature_vec, allowed_actions=allowed_actions, error_free=True)
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
#rStart = 0; cStart = 3;
rStart = np.nan; cStart = np.nan;
#rTerminal = 3; cTerminal = 7
rTerminal = np.nan; cTerminal = np.nan
#rTerminal = 7; cTerminal = 7
#A_effect_vec = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
A_effect_vec = [[0, 1], [0, -1],[1, 0], [-1, 0]]
wind_vec = np.zeros((nC))
#wind_vec[np.array([3, 4, 5, 6])] = 1
pit_vec = np.array([])
#pit_vec = np.array([[0, 4], [0, 5], [0, 6], [4, 4], [4, 5], [4, 6]])
pit_prob = 0.5
pit_punishment = -1
backtrack_punishment = -1
terminal_reward = 0
wall_vec = np.array([])
# wall_vec = np.array([[4, 4], [5, 4], [6, 4], [7, 4], [8, 4]]) # , [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3]
environment = Environment(nR, nC, rStart, cStart, rTerminal, cTerminal, A_effect_vec)
environment.define_specifics(wind_vec, pit_vec, pit_prob, wall_vec, pit_punishment, backtrack_punishment, terminal_reward, [False, True, True])

obs_ind = environment.get_observables_indices()

sarsa = SARSA(environment.nFeatures, environment.nA)
sarsa.lambda0 = 0

max_episode_length = 1e3
sim = Simulation(max_episode_length)

sarsa = sim.train(1e5, environment, sarsa)
route = sim.test(environment, sarsa)
sim.plots(environment, sarsa, route)