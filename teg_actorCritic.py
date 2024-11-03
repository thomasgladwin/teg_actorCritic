import numpy as np

class Environment:
    def __init__(self, A_effect_vec, observable_features=(True, False, False), no_backtrack=True, MapString='', random_pit_prob=0.0, nR=0, nC=0, rStart=0, cStart=0, rTerm=0, cTerm=0, pit_vec=np.array([]), wall_vec=np.array([])):
        self.A_effect_vec = A_effect_vec
        self.nA = len(self.A_effect_vec)
        if not len(MapString) == 0:
            # Define world using a MapString
            MapLines = [r for r in MapString.split('\n') if len(r) > 0]
            tileCodes = np.array([np.fromstring(r, dtype=int, sep=' ') for r in MapLines])
            self.nR = tileCodes.shape[0]
            self.nC = tileCodes.shape[1]
            self.rStart = np.where(tileCodes == 1)[0][0]
            self.cStart = np.where(tileCodes == 1)[1][0]
            self.rTerm0 = np.where(tileCodes == 2)[0][0]
            self.cTerm0 = np.where(tileCodes == 2)[1][0]
            self.pit_vec, self.wall_vec = self.tileCodes_to_vectors(tileCodes)
        else:
            # Define world directly using variables
            self.nR = nR
            self.nC = nC
            self.rStart = rStart
            self.cStart = cStart
            self.rTerm0 = rTerm
            self.cTerm0 = cTerm
            self.pit_vec = pit_vec
            self.wall_vec = wall_vec
        self.s_r = 0
        self.s_c = 0
        self.rTerm = self.rTerm0
        self.cTerm = self.cTerm0
        self.pit_map = np.zeros((self.nR, self.nC))
        self.random_pit_prob = random_pit_prob
        self.create_wall_map()
        self.no_backtrack = no_backtrack
        self.mem_length = self.nR * self.nC
        self.memory = []
        self.observable_features = observable_features  # Coordinates; local pits; goal direction
        feature_vec, allowed_actions = self.state_to_features()
        self.nFeatures = int(np.prod(feature_vec.shape))
        self.run_checks()
    def run_checks(self):
        if self.rTerm0 == self.rStart and self.cTerm0 == self.cStart and not (np.isnan(self.rStart) or np.isnan(self.cStart)):
            raise Exception("Invalid environment definition: start and terminal points are the same.")
    def tileCodes_to_vectors(self, tileCodes):
        pit_vec = np.array([z for z in zip(np.where(tileCodes == 3)[0], np.where(tileCodes == 3)[1])])
        wall_vec = np.array([z for z in zip(np.where(tileCodes == 4)[0], np.where(tileCodes == 4)[1])])
        return pit_vec, wall_vec
    def create_wall_map(self):
        self.wall_map = np.zeros((self.nR, self.nC))
        for i_wall in range(self.wall_vec.shape[0]):
            self.wall_map[self.wall_vec[i_wall][0]][self.wall_vec[i_wall][1]] = 1
    def set_rewards(self, pit_punishment=-1, backtrack_punishment=-1, off_grid_punishment=-1, terminal_reward=0):
        self.pit_punishment = pit_punishment
        self.backtrack_punishment = backtrack_punishment
        self.off_grid_punishment = off_grid_punishment
        self.terminal_reward = terminal_reward
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
        for i_pit in range(self.pit_vec.shape[0]):
            r = self.pit_vec[i_pit][0]
            c = self.pit_vec[i_pit][1]
            if True or (not (np.abs(r - self.rTerm) <= 1 and np.abs(c - self.cTerm) <= 1)):
                self.pit_map[r][c] = 1
        for r in range(self.nR):
            for c in range(self.nC):
                die = np.random.rand()
                if die < self.random_pit_prob:
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
        X = X.reshape(self.nR * self.nC)
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
        X = []
        dr = 0
        if self.s_r - self.rTerm > 0:
            dr = 1
        elif self.s_r - self.rTerm < 0:
            dr = -1
        dc = 0
        if self.s_c - self.cTerm > 0:
            dc = 1
        elif self.s_c - self.cTerm < 0:
            dc = -1
        for r in range(-1, 2):
            for c in range(-1, 2):
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
        f = e + 9
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
        X = X.reshape((len(X), 1))
        allowed_actions = np.array([])
        for a in range(len(self.A_effect_vec)):
            new_s_r = self.s_r + self.A_effect_vec[a][0]
            new_s_c = self.s_c + self.A_effect_vec[a][1]
            if (new_s_r >= 0 and new_s_r < self.nR) and (new_s_c >= 0 and new_s_c < self.nC) and not self.f_into_wall(new_s_r, new_s_c):
                if ((self.no_backtrack == False) or (not (new_s_r, new_s_c) in self.memory)):
                    allowed_actions = np.append(allowed_actions, a)
        if len(allowed_actions) == 0:
            for a in range(len(self.A_effect_vec)):
                new_s_r = self.s_r + self.A_effect_vec[a][0]
                new_s_c = self.s_c + self.A_effect_vec[a][1]
                if (new_s_r >= 0 and new_s_r < self.nR) and (new_s_c >= 0 and new_s_c < self.nC) and not self.f_into_wall(new_s_r, new_s_c):
                    allowed_actions = np.append(allowed_actions, a)
        if len(allowed_actions) == 0:
            for n in range(100):
                print('XXX');
        return X, allowed_actions
    def respond_to_action(self, a):
        off_grid = False
        new_s_r = self.s_r + self.A_effect_vec[a][0]
        new_s_c = self.s_c + self.A_effect_vec[a][1]
        if new_s_c < 0 or new_s_c >= self.nC:
            off_grid = True
        new_s_c = np.min([self.nC - 1, np.max([0, new_s_c])])
        if new_s_r < 0 or new_s_r >= self.nR:
            off_grid = True
        self.s_r = np.min([self.nR - 1, np.max([0, new_s_r])])
        self.s_c = np.min([self.nC - 1, np.max([0, new_s_c])])
        r = -1
        terminal = False
        if self.f_terminal():
            r = self.terminal_reward
            terminal = True
        else:
            if off_grid:
                r = r + self.off_grid_punishment
            if self.f_pit():
                r = r + self.pit_punishment
            if self.f_backtracking():
                r = r + self.backtrack_punishment
        if self.mem_length > 0:
            self.memory.pop(0)
            self.memory.append((self.s_r, self.s_c))
        return r, terminal
