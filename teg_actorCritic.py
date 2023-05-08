# Actor-critic model with a linear value function on indicator features and linear, softmax action preferences
# Continuing version.
# From Sutton & Barto chapter 13.
import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, nR, nC, rStart, cStart, rTerm, cTerm, A_effect_vec):
        self.nR = nR
        self.nC = nC
        self.nStates = self.nR * self.nC
        self.A_effect_vec = A_effect_vec
        self.nA = len(self.A_effect_vec)
        self.rStart = rStart
        self.cStart = cStart
        self.rTerm = rTerm
        self.cTerm = cTerm
        self.init_episode()
    def define_specifics(self, wind_vec):
        self.wind_vec = wind_vec
    def init_episode(self):
        self.s_r = self.rStart
        self.s_c = self.cStart
    def f_terminal(self):
        if self.s_r == self.rTerm and self.s_c == self.cTerm:
            return True
        else:
            return False
    def state_to_features(self):
        X = np.zeros((self.nR, self.nC))
        X[self.s_r, self.s_c] = 1
        X = X.reshape(self.nStates, 1)
        return X
    def respond_to_action(self, a):
        new_s_r = self.s_r + self.A_effect_vec[a][0]
        new_s_c = self.s_c + self.A_effect_vec[a][1]
        new_s_c = np.min([self.nC - 1, np.max([0, new_s_c])])
        new_s_r = new_s_r + self.wind_vec[new_s_c]
        new_s_r = np.min([self.nR - 1, np.max([0, new_s_r])])
        self.s_r = new_s_r
        self.s_c = new_s_c
        if self.f_terminal():
            r = 0
            terminal = True
            self.init_episode()
        else:
            r = -1
            terminal = False
        return r, terminal

class Critic:
    def __init__(self, nFeatures):
        self.w = np.zeros((nFeatures, 1))
        self.z_w = np.zeros((nFeatures, 1))
        self.mean_r = 0
        self.alpha0_w = 0.5
        self.alpha0_mean_r = 0.1
        self.lambda0_w = 0.5
    def get_v(self, feature_vec):
        return np.sum(feature_vec * self.w)
    def delta_v(self, feature_vec):
        return feature_vec
    def update(self, r, feature_vec, feature_vec_new):
        self.delta0 = r - self.mean_r + self.get_v(feature_vec_new) - self.get_v(feature_vec)
        self.mean_r = self.mean_r + self.alpha0_mean_r * self.delta0
        self.z_w = self.lambda0_w * self.z_w + self.delta_v(feature_vec)
        self.w = self.w + self.alpha0_w * self.delta0 * self.z_w
    def get_delta(self):
        return self.delta0

class Actor:
    def __init__(self, nFeatures, nA):
        self.nFeatures = nFeatures
        self.nA = nA
        self.theta0 = np.zeros((self.nFeatures, self.nA))
        self.z_theta = np.zeros((self.nFeatures, self.nA))
        self.alpha0_theta = 0.5
        self.lambda0_theta = 0.5
        self.a = 0
    def policy_prob(self, feature_vec, b):
        prefs = np.exp(np.dot(np.transpose(feature_vec), self.theta0))
        prob = prefs[0, b] / np.sum(prefs)
        return prob
    def act_on_policy(self, feature_vec):
        probs = np.array([])
        for b in range(self.nA):
            prob = self.policy_prob(feature_vec, b)
            probs = np.append(probs, prob)
        self.a = np.random.choice(range(self.nA), p=probs)
        return self.a
    def delta_ln_pi(self, feature_vec):
        term1 = np.zeros((self.nFeatures, self.nA))
        iState = np.where(feature_vec == 1)[0][0]
        term1[iState, self.a] = 1
        term2 = np.zeros((self.nFeatures, self.nA))
        for b in range(self.nA):
            tmp = np.zeros((self.nFeatures, self.nA))
            tmp[iState, b] = 1
            term2 = term2 + self.policy_prob(feature_vec, b) * tmp[iState, b]
        return term1 - term2
    def update(self, delta0, feature_vec):
        self.z_theta = self.lambda0_theta * self.z_theta + self.delta_ln_pi(feature_vec)
        self.theta0 = self.theta0 + self.alpha0_theta * delta0 * self.z_theta

class Simulation:
    def __init__(self):
        self.ep_len = np.array([])
        pass
    def train(self, nEpisodes, environment, critic, actor):
        self.ep_len = np.array([])
        iEpisode = 0
        t_ep = 0
        while iEpisode < nEpisodes:
            print(iEpisode, ', (', environment.s_r, ', ', environment.s_c, ')', sep='')
            feature_vec = environment.state_to_features()
            a = actor.act_on_policy(feature_vec)
            r, terminal = environment.respond_to_action(a)
            if terminal == True:
                self.ep_len = np.append(self.ep_len, t_ep)
                t_ep = 0
                iEpisode = iEpisode + 1
            feature_vec_new = environment.state_to_features()
            critic.update(r, feature_vec, feature_vec_new)
            delta0 = critic.get_delta()
            actor.update(delta0, feature_vec)
            t_ep = t_ep + 1
        return (critic, actor)
    def test(self, environment, actor):
        environment.init_episode()
        terminal = False
        t = 0
        while not terminal:
            feature_vec = environment.state_to_features()
            a = actor.act_on_policy(feature_vec)
            print(t, environment.s_r, environment.s_c)
            r, terminal = environment.respond_to_action(a)
            t = t + 1
    def plots(self, environment, critic, actor):
        W = critic.w.reshape(environment.nR, environment.nC)
        T = actor.theta0
        figs, ax = plt.subplots(3, 1)
        ax[0].plot(self.ep_len)
        ax[1].pcolormesh(W)
        ax[2].pcolormesh(T)
        plt.show()

# Inits
A_effect_vec = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
wind_vec = np.array([0, 0, -1, -2, -1, 0, 2, 2, 1, 0, 0, 0])
environment = Environment(5, 12, 3, 0, 3, 7, A_effect_vec)
environment.define_specifics(wind_vec)
critic = Critic(environment.nStates)
actor = Actor(environment.nStates, environment.nA)
sim = Simulation()

critic, actor = sim.train(1000, environment, critic, actor)
sim.test(environment, actor)
sim.plots(environment, critic, actor)
