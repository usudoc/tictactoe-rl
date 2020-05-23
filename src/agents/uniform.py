import numpy as np

from .agent import Agent


# 一様ランダムに行動を選択するエージェント
class Uniform(Agent):
    def __init__(self, n_state, n_action):
        super().__init__(n_state, n_action)

    # 行動の選択
    def action(self, cur_state, choosable_action):
        return np.random.choice(choosable_action)
