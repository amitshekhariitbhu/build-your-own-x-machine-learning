import numpy as np

# Monte Carlo Tree Search (MCTS) from scratch, played on Tic-Tac-Toe.
# The "synthetic data with planted structure" is a two-player game whose
# latent optimal strategy MCTS must recover using only random rollouts:
# UCB1 selection -> expansion -> random simulation -> backpropagation.
# Players are +1 (X, moves first) and -1 (O). No game logic library is used.

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
         (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
         (0, 4, 8), (2, 4, 6)]              # diagonals


def legal_moves(board):
    return [i for i in range(9) if board[i] == 0]


def apply_move(board, m, player):
    b = list(board)
    b[m] = player
    return tuple(b)


def winner(board):
    # Returns +1 / -1 if that player has a line, 0 for a full-board draw,
    # or None if the game is still in progress.
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0 if 0 not in board else None


class Node:
    # A game state in the search tree. `player` is who moves at this node;
    # the player who moved INTO it is -player, and W/N are scored from that
    # mover's perspective (that is what UCB1 needs for the parent's choice).
    def __init__(self, board, player, parent, move):
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move
        self.children = []
        self.untried = legal_moves(board) if winner(board) is None else []
        self.N = 0
        self.W = 0.0

    def ucb(self, c):
        # Exploitation (mean value) + exploration bonus.
        return self.W / self.N + c * np.sqrt(np.log(self.parent.N) / self.N)


def rollout(board, player):
    # Simulate a uniformly random playout to a terminal state.
    w = winner(board)
    while w is None:
        ms = legal_moves(board)
        board = apply_move(board, ms[np.random.randint(len(ms))], player)
        player = -player
        w = winner(board)
    return w


def mcts_move(board, player, n_iter=200, c=1.4):
    root = Node(board, player, None, None)
    for _ in range(n_iter):
        node = root
        # 1) Selection: descend fully-expanded nodes by best UCB score.
        while not node.untried and node.children:
            node = max(node.children, key=lambda ch: ch.ucb(c))
        # 2) Expansion: add one child for an untried move.
        if node.untried:
            m = node.untried.pop(np.random.randint(len(node.untried)))
            child = Node(apply_move(node.board, m, node.player),
                         -node.player, node, m)
            node.children.append(child)
            node = child
        # 3) Simulation: random rollout from the new node.
        result = rollout(node.board, node.player)
        # 4) Backpropagation: credit each node from its mover's perspective.
        while node is not None:
            node.N += 1
            mover = -node.player
            if node.move is not None:
                node.W += 1.0 if result == mover else 0.5 if result == 0 else 0.0
            node = node.parent
    # Robust choice: the most-visited child.
    return max(root.children, key=lambda ch: ch.N).move


def random_agent(board, player):
    ms = legal_moves(board)
    return ms[np.random.randint(len(ms))]


def mcts_agent(n_iter):
    return lambda board, player: mcts_move(board, player, n_iter)


def play_game(agent_x, agent_o):
    board, player = (0,) * 9, 1
    while True:
        w = winner(board)
        if w is not None:
            return w
        move = (agent_x if player == 1 else agent_o)(board, player)
        board = apply_move(board, move, player)
        player = -player


def non_loss_rate(agent_x, agent_o, games):
    # Fraction of games the first player (X) does not lose (win or draw).
    return np.mean([play_game(agent_x, agent_o) != -1 for _ in range(games)])


if __name__ == "__main__":
    np.random.seed(0)

    # --- Exact, hand-verifiable tactical checks (planted forced moves) ---
    # X (to move) has 0,1; completing at cell 2 wins the top row.
    win_pos = (1, 1, 0, -1, -1, 0, 0, 0, 0)
    found_win = mcts_move(win_pos, 1, n_iter=400) == 2

    # O threatens 0,1; X (to move) must block at cell 2 or lose next turn.
    block_pos = (-1, -1, 0, 1, 0, 0, 1, 0, 0)
    found_block = mcts_move(block_pos, 1, n_iter=400) == 2

    # --- Win-rate tournament vs a random opponent (X plays first) ---
    games = 60
    base = non_loss_rate(random_agent, random_agent, games)      # baseline
    mcts = mcts_agent(n_iter=150)
    score = non_loss_rate(mcts, random_agent, games)             # MCTS as X

    print("Tic-Tac-Toe, first player (X) non-loss rate vs random opponent")
    print("Random X vs Random O -> non-loss rate: %.2f  (baseline)" % base)
    print("MCTS   X vs Random O -> non-loss rate: %.2f" % score)
    print("Beats random baseline:", bool(score > base + 0.15))
    print("Tactical win move found :", bool(found_win))
    print("Tactical block move found:", bool(found_block))
    print("All checks passed:",
          bool(score > base + 0.15 and found_win and found_block))
