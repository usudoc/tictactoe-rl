from .agent import Agent


# MiniMax法
class MiniMax(Agent):
    def __init__(self, n_state, n_action):
        super().__init__(n_state, n_action)

    def action(self, state):
        self.minimax(True, 0)

    def minimax(self, my_turn=True, depth=0):
        self_value = None  # 自身の評価値
        child_value = None  # 子ノードから伝わる評価値
        best_square = None  # 置く位置


        if depth == 0:
            return 0
