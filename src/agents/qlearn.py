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


# Q学習(方策オフ型TD制御アルゴリズム)
class QLearn(Agent):
    def __init__(self, n_state, n_action, alpha, gamma, policy, eps):
        super().__init__(n_state, n_action)
        self.alpha = alpha  # 学習率
        self.gamma = gamma  # 割引率
        self.policy = policy  # 行動を決める方策
        self.eps = eps
        self._eps = eps
        self.delta_eps = 0.0  # epsilon減衰時の減衰値

        self.name = 'QLearn(α={})'.format(self.alpha)

    # 行動の選択
    def action(self, state, choosable_action):
        # greedy方策
        if self.policy == 'greedy':
            return choosable_action[greedy(self.q_table[state, choosable_action])]
        # epsilon-greedy方策
        elif self.policy == 'eps-greedy':
            return choosable_action[eps_greedy(self.q_table[state, choosable_action], self.eps)]

    # パラメータの更新
    def update(self, cur_state, action, reward, next_state):
        super().update()
        # 勝敗が決まって報酬が得られた時
        if reward != 0:
            td_error = reward - self.q_table[cur_state, action]
        # ゲーム継続または引き分け時
        else:
            td_error = reward + self.gamma * np.amax(self.q_table[next_state]) - self.q_table[cur_state, action]
        self.q_table[cur_state, action] += self.alpha * td_error
