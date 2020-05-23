import numpy as np

from .agent import Agent


def greedy(value_list):
    max_values = np.where(value_list == np.amax(value_list))
    return np.random.choice(max_values[0])


def eps_greedy(value_list, eps):
    if np.random.rand() < eps:  # greedy
        return np.random.randint(0, len(value_list))
    else:  # 一様ランダム
        return greedy(value_list)


# Sarsa(方策オン型TD制御アルゴリズム)
class Sarsa(Agent):
    def __init__(self, n_state, n_action, alpha, gamma, policy, eps):
        super().__init__(n_state, n_action)
        self.alpha = alpha  # 学習率
        self.gamma = gamma  # 割引率
        self.policy = policy  # 行動を決める方策
        self.eps = eps
        self._eps = eps
        self.delta_eps = 0.0  # epsilon減衰時の減衰値
        self.cur_action = None

        self.name = 'Sarsa(α={})'.format(self.alpha)

    # 行動の選択
    def action(self, state):
        # 最初は方策に従って行動を選択
        if self.step == 0:
            return self._select_action(state)
        # 2回目以降は推定方策に従って行動を選択
        else:
            return self.cur_action

    def _select_action(self, state):
        # greedy方策
        if self.policy == 'greedy':
            return greedy(self.q_table[state])
        # epsilon-greedy方策
        elif self.policy == 'eps-greedy':
            return eps_greedy(self.q_table[state], self.eps)

    # パラメータの更新
    def update(self, cur_state, action, reward, next_state):
        super().update()
        # 実際の方策に従って次状態における行動を選択
        next_action = self._select_action(next_state)
        td_error = reward + self.gamma * self.q_table[next_state, next_action] - self.q_table[cur_state, action]
        self.q_table[cur_state, action] += self.alpha * td_error
        self.cur_action = next_action
