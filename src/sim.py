import numpy as np
import matplotlib.pyplot as plt

from uniform import Uniform


class Sim(object):
    def __init__(self):
        pass

    # シミュレーションの実行
    def simulation(self, env, agt_list, agt_enemy, n_sims=1, n_epis=100, max_turns=9):
        ev = env  # 環境
        n_agt = len(agt_list)  # 全エージェント数

        turn_mode = 0  # 0:順番は常に先手 1:順番は常に後手 2:順番は交互に先手を取る
        players = [None, None]  # 実際に行動するエージェントリスト
        players[1] = agt_enemy  # 敵となるエージェント(Player2)

        name_list = []
        won_counts = np.zeros((n_agt, n_epis))  # エージェントの勝利数
        draw_counts = np.zeros((n_agt, n_epis))  # 引き分け数
        won_counts_100 = np.zeros((n_agt, int(n_epis / 100)))  # 100回ごとの勝利数

        for agt_idx, agt in enumerate(agt_list):
            print(agt.name)
            name_list.append(agt.name)
            players[0] = agt  # player1にエージェントを格納

            for sim in range(n_sims):
                # print('sim :', sim + 1)
                agt.initialize()
                sum_won_counts = 0
                sum_draw_counts = 0

                for epi in range(n_epis):
                    print('epi : ', epi)
                    ev.initialize()  # 環境の初期化

                    pre_player = None
                    # 先手/後手を与えられた引数に従って決める
                    if turn_mode == 0 or turn_mode == 1:  # 常に先手/後手
                        pre_player = turn_mode + 1
                    elif turn_mode == 2:  # 交互に先手となる
                        pre_player = epi % 2 + 1
                    cur_player = pre_player

                    for turn in range(max_turns):
                        cur_state = ev.board_to_state()  # 現状態の取得
                        choosable_actions = ev.get_vacant_squares()  # 選択可能な行動を取得
                        # print(choosable_actions)
                        action = players[cur_player - 1].action(cur_state, choosable_actions)  # 行動の選択
                        # print(cur_player, ':', action)
                        reward, next_state = ev.step(action, cur_player)  # 選択された行動の反映
                        # エージェントのパラメータ更新

                        # 勝敗が決まった時
                        if reward == 1 or reward == -1:
                            players[0].update(cur_state, action, reward, next_state)
                            break
                        # 勝敗が決まらない
                        elif reward == 0:
                            # 引き分け
                            if turn == max_turns - 1:
                                players[0].update(cur_state, action, reward, next_state)
                                pass
                            # ゲーム続行
                            else:
                                players[0].update(cur_state, action, reward, next_state)
                                cur_player = (pre_player + turn) % 2 + 1
                                pass

                    # 1試合の終了時
                    print('reward : ', reward)
                    # env.show_board()  # 盤面状況を図示
                    # エージェントの勝利数をカウント(エージェントは1, ランダムは2)
                    sum_won_counts += 1 if reward == 1 else 0
                    # 勝者が存在していなければ引き分けをカウント
                    sum_draw_counts += 1 if reward == 0 else 0
                    won_counts[agt_idx, epi] += sum_won_counts
                    draw_counts[agt_idx, epi] += sum_draw_counts
                    won_counts_100[agt_idx, int(epi / 100)] += 1 if reward == 1 else 0

        won_counts /= n_sims
        draw_counts /= n_sims
        won_counts_100 /= n_sims
        print('Win : ', won_counts[:, -1])
        print('Lose : ', n_epis - won_counts[:, -1] - draw_counts[:, -1])
        print('Draw : ', draw_counts[:, -1])
        print('Win rate : {}%'.format(won_counts_100[:, -1] / 100))

        return name_list, won_counts, draw_counts, won_counts_100

    # シミュレーションとプロットの実行
    def run(self, env, agt_list, agt_enemy, n_sims=1, n_epis=100, max_turns=9):
        name_list, won_count, draw_count, won_counts_100 = \
            self.simulation(env, agt_list, agt_enemy, n_sims, n_epis, max_turns)
        # シミュレーション結果をプロット
        self.plot(won_count, title='Result', xlabel='Episodes', ylabel='Won_counts', name_list=name_list)
        self.plot(draw_count, title='Result', xlabel='Episodes', ylabel='Draw_counts', name_list=name_list)
        self.plot(won_counts_100, title='Result', xlabel='Episodes', ylabel='Win_rate', name_list=name_list)

    # データをグラフにプロット
    def plot(self, data_list, title, xlabel, ylabel, name_list):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        for i, data in enumerate(data_list):
            ax.plot(data, label=name_list[i], linewidth=1.5, alpha=0.8)

        # ax.title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper left', fontsize=14)
        ax.grid(axis='y')

        plt.show()