# Actor-critic model with a linear value and preference functions
# Episodic.
# From Sutton & Barto chapter 13.
import numpy as np
import matplotlib.pyplot as plt
import Environment
import importlib
importlib.reload(Environment)

class Critic:
    def __init__(self, nFeatures):
        self.w = np.zeros((nFeatures, 1))
        self.z = np.zeros((nFeatures, 1))
        self.alpha0 = 0.1
        self.lambda0 = 0.5
        self.gamma0 = 0.1
    def get_v(self, feature_vec):
        return np.sum(feature_vec * self.w)
    def delta_v(self, feature_vec):
        return feature_vec
    def update(self, r, feature_vec, feature_vec_new, terminal):
        if not terminal:
            self.delta0 = r + self.gamma0 * self.get_v(feature_vec_new) - self.get_v(feature_vec)
        else:
            self.delta0 = r - self.get_v(feature_vec)
        self.z = self.lambda0 * self.z + self.delta_v(feature_vec)
        self.w = self.w + self.alpha0 * self.delta0 * self.z
    def get_delta(self):
        return self.delta0

class Actor:
    def __init__(self, nFeatures, nA, action_error_prob=0.1):
        self.nFeatures = nFeatures
        self.nA = nA
        self.action_error_prob = action_error_prob
        self.theta0 = np.zeros((self.nFeatures, self.nA))
        self.z = np.zeros((self.nFeatures, self.nA))
        self.alpha0 = 0.5 # 0.1
        self.lambda0 = 0.5
        self.gamma0 = 0.5
        self.I = 1
        self.a = 0
    def policy_prob(self, feature_vec, b):
        c = np.max(np.dot(np.transpose(feature_vec), self.theta0))
        prefs = np.exp(np.dot(np.transpose(feature_vec), self.theta0) - c)
        prob = prefs[0][b] / np.sum(prefs)
        return prob
    def act_on_policy_q(self, feature_vec, allowed_actions=[], error_free=True):
        if len(allowed_actions) == 0:
            allowed_actions = np.array(range(self.nA))
        allowed_actions = allowed_actions.astype(int)
        action_error_rnd = np.random.rand()
        if action_error_rnd < self.action_error_prob and error_free == False:
            self.a = np.random.choice(allowed_actions)
        else:
            sa_q = np.array([])
            for b in allowed_actions:
                q = np.dot(feature_vec, self.theta0[:, b])
                sa_q = np.append(sa_q, q)
            self.a = allowed_actions[np.argmax(sa_q)]
        return self.a
    def act_on_policy_softmax(self, feature_vec, allowed_actions=[], error_free=True):
        if len(allowed_actions) == 0:
            allowed_actions = np.array(range(self.nA))
        allowed_actions = allowed_actions.astype(int)
        action_error_rnd = np.random.rand()
        if action_error_rnd < self.action_error_prob and error_free == False:
            print('random action')
            self.a = np.random.choice(allowed_actions)
        else:
            probs = np.array([])
            for b in range(self.nA):
                prob = self.policy_prob(feature_vec, b)
                probs = np.append(probs, prob)
            probs = probs[allowed_actions]
            if np.any(np.isnan(probs)) or np.sum(probs) == 0:
                #print('X-X-X\nX-X-X\nIllegal probs: ', probs, ', theta0: ', self.theta0, 'z: ', self.z, '\nX-X-X\nX-X-X\n')
                probs = np.ones(probs.shape) / len(probs) # If need to choose between effectively 0-prob allowed actions
            else:
                probs = probs / np.sum(probs)
            self.a = np.random.choice(allowed_actions, p=probs)
        return self.a
    def act_on_policy(self, feature_vec, allowed_actions=[], error_free=True):
        self.a = self.act_on_policy_softmax(feature_vec, allowed_actions, error_free)
        return self.a
    def delta_ln_pi(self, feature_vec):
        term1 = np.zeros((self.nFeatures, self.nA))
        iStates = np.where(feature_vec == 1)[0]
        term1[iStates, self.a] = 1
        term2 = np.zeros((self.nFeatures, self.nA))
        for b in range(self.nA):
            tmp = np.zeros((self.nFeatures, self.nA))
            tmp[iStates, b] = 1
            term2 = term2 + self.policy_prob(feature_vec, b) * tmp
        return term1 - term2
    def update(self, delta0, feature_vec):
        delta_this = self.delta_ln_pi(feature_vec)
        self.z = self.lambda0 * self.z + self.I * delta_this
        self.theta0 = self.theta0 + self.alpha0 * delta0 * self.z
        self.I = self.I * self.gamma0

class ActorCritic:
    def __init__(self, nFeatures, nA):
        self.critic = Critic(nFeatures)
        self.actor = Actor(nFeatures, nA)
    def init_episode(self):
        self.critic.z = 0 * self.critic.z
        self.actor.z = 0 * self.actor.z
        self.actor.I = 1

class Simulation:
    def __init__(self, max_episode_length):
        self.ep_len = np.array([])
        self.max_episode_length = max_episode_length
        pass
    def train(self, nEpisodes, environment, agent):
        environment.init_episode()
        agent.init_episode()
        self.ep_len = np.array([])
        iEpisode = 0
        t_ep = 0
        while iEpisode < nEpisodes:
            print(iEpisode, '. ', end='', sep='')
            print('(', environment.s_r, ', ', environment.s_c, '). ', end='', sep='')
            feature_vec, allowed_actions = environment.state_to_features()
            print(allowed_actions, '. ', sep='', end='')
            a = agent.actor.act_on_policy(feature_vec, allowed_actions=allowed_actions, error_free=False)
            r, terminal = environment.respond_to_action(a)
            feature_vec_new, allowed_actions_new = environment.state_to_features()
            agent.critic.update(r, feature_vec, feature_vec_new, terminal)
            delta0 = agent.critic.get_delta()
            agent.actor.update(delta0, feature_vec)
            print('a = ', a, '. r = ', r, '. delta0 = ', delta0, ', max abs w = ', np.max(np.abs(agent.critic.w)), ', max abs theta0 = ', np.max(np.abs(agent.actor.theta0)), end='\n', sep='')
            if np.isnan(delta0):
                break
            if t_ep > self.max_episode_length:
                print('XXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXX')
                print('Episode failed.')
                print('XXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXX')
                terminal = True
            if terminal == True:
                environment.init_episode()
                agent.init_episode()
                self.ep_len = np.append(self.ep_len, t_ep)
                t_ep = 0
                iEpisode = iEpisode + 1
            t_ep = t_ep + 1
        return agent
    def test(self, environment, agent):
        environment.init_episode()
        terminal = False
        route = np.array([environment.s_r, environment.s_c])
        t = 0
        while not terminal:
            print(t, ': ', end='')
            feature_vec, allowed_actions = environment.state_to_features()
            a = agent.actor.act_on_policy(feature_vec, allowed_actions=allowed_actions, error_free=True)
            print('(', environment.s_r, environment.s_c, ')')
            r, terminal = environment.respond_to_action(a)
            if not terminal:
                route = np.append(route, [environment.s_r, environment.s_c])
            t = t + 1
        route = route.reshape(int(len(route)/2), 2)
        return route
    def plots(self, environment, agent, route):
        obs_ind = environment.get_observables_indices()
        W = agent.critic.w[obs_ind[0][0]:obs_ind[0][1]].reshape((environment.nR, environment.nC))
        W_local = np.max(agent.critic.w[obs_ind[1][0]:obs_ind[1][1]], axis=1).reshape(3, 3)
        W_goal = np.max(agent.critic.w[obs_ind[2][0]:obs_ind[2][1]], axis=1).reshape(3, 3)
        T_local = agent.actor.theta0[obs_ind[1][0]:obs_ind[1][1], :]
        T_goal = agent.actor.theta0[obs_ind[2][0]:obs_ind[2][1], :]
        figs, ax = plt.subplots(4, 2)
        ax[0, 0].plot(self.ep_len)
        ax[1, 0].pcolormesh(W)
        ax[2, 0].pcolormesh(W_local)
        ax[3, 0].pcolormesh(W_goal)
        ax[0, 1].pcolormesh(environment.pit_map + environment.wall_map * 2)
        if len(route) > 0:
            ax[0, 1].scatter(route[:, 1] + 0.5, route[:, 0] + 0.5)
            ax[0, 1].plot(route[:, 1] + 0.5, route[:, 0] + 0.5)
            ax[0, 1].xaxis.set_ticks(ticks=np.array([range(environment.nC)]).reshape(environment.nC) + 0.5, labels=np.array([range(environment.nC)]).reshape(environment.nC))
            ax[0, 1].yaxis.set_ticks(ticks=np.array([range(environment.nR)]).reshape(environment.nR) + 0.5, labels=np.array([range(environment.nR)]).reshape(environment.nR))
        ax[2,1].pcolormesh(T_local)
        ax[3,1].pcolormesh(T_goal)
        plt.show()

# Inits
nR = 7; nC = 9;
#rStart = 0; cStart = 1;
rStart = np.nan; cStart = np.nan;
#rTerminal = 3; cTerminal = 7
rTerminal = np.nan; cTerminal = np.nan
#rTerminal = 7; cTerminal = 7
A_effect_vec = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
#A_effect_vec = [[0, 1], [0, -1],[1, 0], [-1, 0]]
wind_vec = np.zeros((nC))
#wind_vec[np.array([3, 4, 5, 6])] = 1
pit_vec = np.array([])
#pit_vec = np.array([[0, 4], [0, 5], [0, 6], [4, 4], [4, 5], [4, 6]])
pit_prob = 0.2
pit_punishment = -1
backtrack_punishment = 0
off_grid_punishment = -1
terminal_reward = 0
wall_vec = np.array([])
# wall_vec = np.array([[4, 4], [5, 4], [6, 4], [7, 4], [8, 4]]) # , [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3]

environment = Environment.Environment(nR, nC, rStart, cStart, rTerminal, cTerminal, A_effect_vec)
environment.define_specifics(wind_vec, pit_vec, pit_prob, wall_vec, pit_punishment, backtrack_punishment, off_grid_punishment, terminal_reward, [False, True, True])
environment.mem_length = 2

agent = ActorCritic(environment.nFeatures, environment.nA)
agent.critic.lamba0 = 0
agent.actor.lamba0 = 0

max_episode_length = 1e6
sim = Simulation(max_episode_length)

agent = sim.train(1e4, environment, agent)
route = sim.test(environment, agent)
sim.plots(environment, agent, route)

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
route = sim.test(environment, agent)
sim.plots(environment, agent, route)
