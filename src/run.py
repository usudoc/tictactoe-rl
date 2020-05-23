from .sim import Sim
from .environments.tictactoe import Tictactoe
from .agents.uniform import Uniform
from .agents.qlearn import QLearn


N_SIMS = 1  # シミュレーション回数
N_EPIS = 500000  # ゲーム回数
MAX_TURNS = 9  # 最大ターン数

sim = Sim()
ev = Tictactoe()  # 環境オブジェクト
n_state = len(ev.SQUARE_TYPES) ** ev.BOARD.size  # 全状態数(3**9=19683)
n_action = ev.BOARD.size  # 全行動数(9)
agt_enemy = Uniform(n_action, n_action)  # 敵となるエージェント

# エージェントリスト
ag_list = [
  # Sarsa(n_state, n_action, alpha=0.1, gamma=1.0, policy='eps-greedy', eps=0.1),
  QLearn(n_state, n_action, alpha=0.2, gamma=0.8, policy='eps-greedy', eps=0.1)
]

sim.run(env=ev, agt_list=ag_list, agt_enemy=agt_enemy, n_sims=N_SIMS, n_epis=N_EPIS, max_turns=MAX_TURNS)
