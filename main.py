from collections import deque
import time


absolute_score = 1000

class Position:
    def __init__(self, board, move, score=None):
        self.board = board
        self.move = move
        self.winner = '_'
        self.score = score
        self.boardList = {
            'W': {
                'K': [(0, 3)],
                'Q': [(0, 4)],
                'R': [(0, 0), (0, 7)],
                'B': [(0, 2), (0, 5)],
                'N': [(0, 1), (0, 6)],
                'P': [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]
            },
            'B': {
                'K': [(7, 3)],
                'Q': [(7, 4)],
                'R': [(7, 0), (7, 7)],
                'B': [(7, 2), (7, 5)],
                'N': [(7, 1), (7, 6)],
                'P': [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7)]
            }
        }

    def __getitem__(self, item):
        return item

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


def coords_to_square(i, j):
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    return letters[j] + (str) (i + 1)


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


def pieces_score(piece):
    scores = {
        'K': 10000,
        'Q': 9,
        'R': 5,
        'N': 3,
        'B': 3,
        'P': 1
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
                score += pieces_score(piece)
            else:
                score -= pieces_score(piece)

    return score


def minimax_with_tree_generation(position, depth, alpha=float('-inf'), beta=float('inf'),
                                 isRoot=False):

    global absolute_score

    score = position_score(position)

    if score > absolute_score:
        position.winner = 'W'  # White wins
    elif score < -absolute_score:
        position.winner = 'B'  # Black wins
    elif depth == 0 or position.winner != '_':
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
    board = position.board
    won = True
    result = []

    # for i in range(position.m):
    #     for j in range(position.n):
    #         if board[i][j] != color:
    #             continue
    #         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    #             ni, nj = i + dx, j + dy
    #             if 0 <= ni < position.m and 0 <= nj < position.n:
    #                 if board[ni][nj] not in ('_', color):
    #                     won = False
    #                     temp = [row[:] for row in board]
    #                     temp[ni][nj] = color
    #                     temp[i][j] = '_'
    #
    #                     new_position = Position(temp, 'W' if color == 'B' else 'B')
    #
    #                     new_position.score = position_score(new_position)
    #                     result.append(Node(new_position))

    return result, won


def play(position, depth):

    position.printBoard()
    print()

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


