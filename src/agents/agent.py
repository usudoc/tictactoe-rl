import numpy as np


# エージェント
class Agent(object):
    def __init__(self, n_state, n_action):
        self.n_state = n_state
        self.n_action = n_action
        self.q_table = np.zeros((self.n_state, self.n_action))
        self.step = 0

    def initialize(self):
        self.q_table = np.zeros((self.n_state, self.n_action))
        self.step = 0

    def action(self, cur_state):
        pass

    def update(self):
        self.step += 1
