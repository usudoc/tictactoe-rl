import numpy as np

from .environment import Environment


'''
3×3の盤面
----------
|  |  |  |
----------
|  |  |  |
----------
|  |  |  |
----------
'''


# 3目並べ環境
class Tictactoe(Environment):
    SQUARE_TYPES = [
        ' ',  # 0: 何もないマス
        '○',  # 1: マル
        '×'  # 2: バツ
    ]

    BOARD = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    BOARD_RANGE = np.array(BOARD.shape)  # 形は[y, x]([3, 3])

    def __init__(self):
        super().__init__()
        # self.board = self.BOARD
        self.board = self.BOARD.flatten()
        print(self.board)
        self.vacant_squares = np.array([i for i in range(len(self.board))])  # 盤面中の何も置かれていないマスのインデックス
        # self.is_finished(player=1)
        self.cur_player = None

    def initialize(self):
        self.board = self.BOARD
        self.board = self.BOARD.flatten()
        self.vacant_squares = np.array([i for i in range(len(self.board))])  # 盤面中の何も置かれていないマスのインデックス
        self.cur_player = None

    # 行動を行う
    def step(self, action, player):
        self.board[action] = player  # 行動の実行
        # 実行された(置かれた)マスはリストから除く
        self.vacant_squares = \
            self.vacant_squares[~(self.vacant_squares == action)]
        is_finished, winner = self.judge_winner(player)  # 判定
        reward = self.get_reward(winner)  # 判定の結果から報酬を観測
        next_state = self.board_to_state()  # 次状態を取得

        return reward, next_state

    # 報酬の観測
    def get_reward(self, winner):
        reward = 0
        if winner == 1:  # エージェントが勝利した時
            reward = 1
        elif winner == 2:  # エージェントが敗北した時
            reward = -1
        else:  # 勝者が存在せず引き分けor続行時
            reward = 0

        return reward

    # 1ラインが揃っているか判定
    def judge_line(self, val_a, val_b, val_c):
        return val_a == val_b == val_c != 0

    # 勝敗引き分け判定
    def judge_winner(self, player):
        # print('----- judge -----')
        is_finished = True  # 1戦の終了フラグ
        winner = None  # 勝者

        first_list = [0, 3, 6, 0, 1, 2, 0, 2]
        second_list = [1, 4, 7, 3, 4, 5, 4, 4]
        third_list = [2, 5, 8, 6, 7, 8, 8, 6]
        # ラインごとに揃っているか見ていく
        for first, second, third in zip(first_list, second_list, third_list):
            if self.judge_line(self.board[first], self.board[second], self.board[third]):
                winner = player
                is_finished = True
                break
        vacant_square = [x for x in self.board if x == 0]  # 空いているマス候補
        # 空いているマスが無ければ引き分けとして終了フラグを立てる
        if len(vacant_square) == 0:
            is_finished = True

        return is_finished, winner

    # 盤面状況から状態へ変換
    def board_to_state(self):
        state = 0
        for i in range(0, 9):
            if self.board[i] == 1:
                state += 3 ** i * 2
            elif self.board[i] == 2:
                state += 3 ** i * 1
            else:
                state += 3 ** i * 0
        return state

    # 盤面中の何も置かれていないマスのインデックスを返す
    def get_vacant_squares(self):
        return self.vacant_squares

    # 指定された要素番号に駒が置けるかどうかの判定
    def can_place(self, idx):
        if self.board[idx] == 0:
            return True
        else:
            return False

    # 盤面状況を図示
    def show_board(self):
        for i in range(3):
            print('-------')
            for j in range(3):
                print('|{}'.format(self.SQUARE_TYPES[self.board[i*3+j]]), end='')
            print('|')
        print('-------')
