from collections import deque
import time


absolute_score = 100

class Position:
    def __init__(self, board, move, score=None):
        self.board = board
        self.move = move
        self.winner = '_'
        self.score = score

    def printBoard(self):
        for row in self.board:
           print(*row, sep=' ')


class Node:
    def __init__(self, position):
        self.position = position
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def getPosition(self):
        return self.position


def is_winning_position(position):
    pass


def print_tree(tree):
    queue = deque()
    queue.append(tree)

    w = 0
    b = 0

    level = 0
    while queue:
        level_size = len(queue)
        print(f"Level {level}:")

        for _ in range(level_size):
            node = queue.popleft()
            node.position.printBoard()
            print("Winner: ", node.position.winner)
            print("To move: ", node.position.move)
            if node.position.score:
                print("Score: ", node.position.score)
            if node.position.winner == 'B':
                b += 1
            elif node.position.winner == 'W':
                w += 1
            print()
            queue.extend(node.children)

        level += 1

    print("W: ", w, " B: ", b)


def moves(piece_symbol):
    moveset = {
        'K': [[(0, 1), (0, -1), (1, 1), (1, 0), (1, -1), (-1, 1), (-1, 0), (-1, -1)]],
        'Q': [
            [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
            [(-1, -1), (-2, -2), (-3, -3), (-4, -4), (-5, -5), (-6, -6), (-7, -7)],
            [(1, -1), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6), (7, -7)],
            [(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-6, 6), (-7, 7)],
            [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
            [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
            [(-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0), (-6, 0), (-7, 0)],
            [(0, -1), (0, -2), (0, -3), (0, -4), (0, -5), (0, -6), (0, -7)]
        ],
        'R': [
            [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
            [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
            [(-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0), (-6, 0), (-7, 0)],
            [(0, -1), (0, -2), (0, -3), (0, -4), (0, -5), (0, -6), (0, -7)]
        ],
        'B': [
            [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
            [(-1, -1), (-2, -2), (-3, -3), (-4, -4), (-5, -5), (-6, -6), (-7, -7)],
            [(1, -1), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6), (7, -7)],
            [(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-6, 6), (-7, 7)],
        ],
        'N': [
            [(1, 2)],
            [(2, 1)],
            [(-1, 2)],
            [(-2, 1)],
            [(1, -2)],
            [(2, -1)],
            [(-1, -2)],
            [(-2, -1)],
        ]
    }

    return moveset[piece_symbol]


def pieces_score(piece, i, j):
    scores = {
        'WK': 0,
        'BK': 0,
        'WQ': 0,
        'BQ': 0,
        'WR': 0,
        'BR': 0,
        'WN': 0,
        'BN': 0,
        'WB': 0,
        'BB': 0,
        'WP': 0,
        'BP': 0
    }

    return scores[piece]


def position_score(position):
    score = 0

    for i in range(8):
        for j in range(8):
            piece = position[i][j]
            if piece == '_':
                continue
            if piece[0] == 'W':
                score += pieces_score(piece, i, j)
            else:
                score -= pieces_score(piece, i, j)

    return score


def minimax_with_tree_generation(position, depth, alpha=float('-inf'), beta=float('inf'),
                                 isRoot=False):

    global absolute_score

    if position.winner == 'W':
        return None, absolute_score if isRoot else absolute_score  # White wins
    elif position.winner == 'B':
        return None, -absolute_score if isRoot else -absolute_score  # Black wins
    elif depth == 0:
        score = position_score(position)
        return None, score if isRoot else score

    color = position.move
    is_maximizing = (color == 'W')

    next_positions, won = generate_next_possible_positions(position)

    if won:
        position.winner = 'B' if color == 'W' else 'W'
        return None, absolute_score if color == 'W' else -absolute_score

    if not next_positions:
        return None, -absolute_score if color == 'W' else absolute_score

    best_move = None

    if is_maximizing:
        best_score = float('-inf')
        for next_pos in next_positions:
            _, score = minimax_with_tree_generation(next_pos.position, depth - 1, alpha, beta, False)
            if score > best_score:
                best_score = score
                best_move = next_pos
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_move, best_score if isRoot else best_score
    else:
        best_score = float('inf')
        for next_pos in next_positions:
            _, score = minimax_with_tree_generation(next_pos.position, depth - 1, alpha, beta, False)
            if score < best_score:
                best_score = score
                best_move = next_pos
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_move, best_score if isRoot else best_score


def generate_next_possible_positions(position, isCheck=False):

    color = position.move
    board = position.getBoard()
    won = True
    result = []

    for i in range(position.m):
        for j in range(position.n):
            if board[i][j] != color:
                continue
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < position.m and 0 <= nj < position.n:
                    if board[ni][nj] not in ('_', color):
                        won = False
                        temp = [row[:] for row in board]
                        temp[ni][nj] = color
                        temp[i][j] = '_'

                        new_position = Position(temp, 'W' if color == 'B' else 'B')
                        if is_winning_position(new_position):
                            new_position.winner = color

                        new_position.score = position_score(new_position)
                        result.append(Node(new_position))

    return result, won


def play(position, depth):

    position.printBoard()
    print()

    if is_winning_position(position):
        return 'W' if position.move == 'B' else 'B'

    best_move, _ = minimax_with_tree_generation(position, depth,
                                                float('-inf'), float('inf'),
                                                True)

    if best_move is None:
        print("No moves!")
        return 'W' if position.move == 'B' else 'B'

    return best_move.position


def generate_starting_position():
    board = [
        ['WR', 'WN', 'WB', 'WK', 'WQ', 'WB', 'WN', 'WR'],
        ['WP', 'WP', 'WP', 'WP', 'WP', 'WP', 'WP', 'WP'],
        ['_', '_', '_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_', '_', '_'],
        ['BP', 'BP', 'BP', 'BP', 'BP', 'BP', 'BP', 'BP'],
        ['BR', 'BN', 'BB', 'BK', 'BQ', 'BB', 'BN', 'BR']
    ]

    start_pos = Position(board, 'W')
    return start_pos


if __name__ == "__main__":
    depth = 5
    start_pos = generate_starting_position()

    start_time_2 = time.time()
    print("Winner optimized: ", play(start_pos, depth))
    end_time_2 = time.time()
    print(f"Execution time: {end_time_2 - start_time_2:.2f}s")


