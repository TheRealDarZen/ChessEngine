import copy
import time


absolute_score = 1000

class Position:
    def __init__(self, board, move, last_move=None, score=None):
        self.board = board
        self.move = move
        self.last_move = last_move
        self.winner = '_'
        self.score = score
        self.enPassFrom = []
        self.enPassTo = None
        self.boardList = {
            'W': {
                'K': [(0, 3)],
                'Q': [(0, 4)],
                'R': [(0, 0), (0, 7)],
                'B': [(0, 2), (0, 5)],
                'N': [(0, 1), (0, 6)],
                'P': [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]

                # 'P': [(3, 0)],
                # 'Q': [],
                # 'R': [(2, 2)],
                # 'B': [],
                # 'N': [],
            },
            'B': {
                'K': [(7, 3)],
                'Q': [(7, 4)],
                'R': [(7, 0), (7, 7)],
                'B': [(7, 2), (7, 5)],
                'N': [(7, 1), (7, 6)],
                'P': [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7)]

                # 'N': [],
                # 'B': [],
                # 'P': [(3, 1)]
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
    letters = ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']
    return letters[j] + (str) (i + 1)


def moves(piece_symbol):
    moveset = {
        'K': [
            [(0, 1)],
            [(0, -1)],
            [(1, 1)],
            [(1, 0)],
            [(1, -1)],
            [(-1, 1)],
            [(-1, 0)],
            [(-1, -1)]],
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
    costs = {
        'K': 10000,
        'Q': 9,
        'R': 5,
        'N': 3,
        'B': 3,
        'P': 1
    }

    return costs[piece]


def position_score(position):
    score = 0

    # TODO Score

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

    next_positions = generate_next_possible_positions(position)

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


def generate_next_possible_positions(position):

    color = position.move
    board = position.board
    boardList = position.boardList
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

    for piece in boardList[color]:
        for piecePos in boardList[color][piece]:
            if piece != 'P':
                allMoves = moves(piece)
                for line in allMoves:
                    for move in line:
                        opPiece = False
                        di, dj = move
                        ni, nj = piecePos[0] + di, piecePos[1] + dj
                        if 0 <= ni < 8 and 0 <= nj < 8:
                            if board[ni][nj][0] != color:

                                if board[ni][nj] != '_': # Other piece
                                    opPiece = True

                                # Board
                                tempBoard = [row[:] for row in board]
                                tempBoard[ni][nj] = (color + piece)
                                tempBoard[piecePos[0]][piecePos[1]] = '_'

                                # Board List
                                tempBoardList = copy.deepcopy(boardList)
                                tempBoardList[color][piece].remove((piecePos[0], piecePos[1]))
                                tempBoardList[color][piece].append((ni, nj))

                                if opPiece: # remove other piece
                                    tempBoardList[board[ni][nj][0]][board[ni][nj][1]].remove((ni, nj))

                                new_position = Position(tempBoard, 'W' if color == 'B' else 'B', (piece, piecePos[0], piecePos[1], ni, nj))
                                new_position.boardList = tempBoardList
                                new_position.score = position_score(new_position)

                                result.append(Node(new_position))

                                if opPiece:
                                    break

                            else:
                                break

                        else:
                            break

            # Pawns
            else:
                allPossibleMovesTo = []
                tempEnPassFrom = []
                tempEnPassTo = None
                change = False

                # White
                if color == 'W':
                    if board[piecePos[0] + 1][piecePos[1]] == '_':

                        allPossibleMovesTo.append((piecePos[0] + 1, piecePos[1]))

                        if piecePos[0] == 6:
                            change = True

                        if piecePos[0] == 1: # 2nd rank
                            if board[piecePos[0] + 2][piecePos[1]] == '_':
                                allPossibleMovesTo.append((piecePos[0] + 2, piecePos[1]))
                                if piecePos[1] > 0:
                                    tempEnPassFrom.append((piecePos[0] + 2, piecePos[1] - 1))
                                if piecePos[1] < 7:
                                    tempEnPassFrom.append((piecePos[0] + 2, piecePos[1] + 1))
                                tempEnPassTo = (piecePos[0] + 1, piecePos[1])

                    # En Passant
                    if (piecePos[0], piecePos[1]) in position.enPassFrom:
                        allPossibleMovesTo.append(position.enPassTo)

                    # Captures
                    if piecePos[1] > 0:
                        if board[piecePos[0] + 1][piecePos[1] - 1][0] == 'B':
                            allPossibleMovesTo.append((piecePos[0] + 1, piecePos[1] - 1))
                    if piecePos[1] < 7:
                        if board[piecePos[0] + 1][piecePos[1] + 1][0] == 'B':
                            allPossibleMovesTo.append((piecePos[0] + 1, piecePos[1] + 1))

                    # Generate actual moves
                    transformList = ['Q', 'R', 'N', 'B'] if change else ['P']
                    for move in allPossibleMovesTo:
                        ni, nj = move

                        for tp in transformList:
                            # Board
                            tempBoard = [row[:] for row in board]
                            tempBoard[ni][nj] = (color + tp)
                            tempBoard[piecePos[0]][piecePos[1]] = '_'

                            # Board List
                            tempBoardList = copy.deepcopy(boardList)
                            tempBoardList[color][piece].remove((piecePos[0], piecePos[1]))
                            tempBoardList[color][tp].append((ni, nj))

                            if ni - piecePos[0] != 0 and nj - piecePos[1] != 0: # remove other piece
                                if board[ni][nj] != '_':
                                    tempBoardList[board[ni][nj][0]][board[ni][nj][1]].remove((ni, nj))
                                else:
                                    tempBoardList[board[ni - 1][nj][0]][board[ni - 1][nj][1]].remove((ni - 1, nj))
                                    tempBoard[ni - 1][nj] = '_'

                            new_position = Position(tempBoard, 'B',
                                                    (piece, piecePos[0], piecePos[1], ni, nj))
                            new_position.boardList = tempBoardList
                            new_position.enPassFrom = tempEnPassFrom
                            new_position.enPassTo = tempEnPassTo
                            new_position.score = position_score(new_position)

                            result.append(Node(new_position))

                # Black
                else:
                    if board[piecePos[0] - 1][piecePos[1]] == '_':

                        allPossibleMovesTo.append((piecePos[0] - 1, piecePos[1]))

                        if piecePos[0] == 1:
                            change = True

                        if piecePos[0] == 1:  # 2nd rank
                            if board[piecePos[0] - 2][piecePos[1]] == '_':
                                allPossibleMovesTo.append((piecePos[0] - 2, piecePos[1]))
                                if piecePos[1] > 0:
                                    tempEnPassFrom.append((piecePos[0] - 2, piecePos[1] - 1))
                                if piecePos[1] < 7:
                                    tempEnPassFrom.append((piecePos[0] - 2, piecePos[1] + 1))
                                tempEnPassTo = (piecePos[0] - 1, piecePos[1])

                    # En Passant
                    if (piecePos[0], piecePos[1]) in position.enPassFrom:
                        allPossibleMovesTo.append(position.enPassTo)

                    # Captures
                    if piecePos[1] > 0:
                        if board[piecePos[0] - 1][piecePos[1] - 1][0] == 'W':
                            allPossibleMovesTo.append((piecePos[0] - 1, piecePos[1] - 1))
                    if piecePos[1] < 7:
                        if board[piecePos[0] - 1][piecePos[1] + 1][0] == 'W':
                            allPossibleMovesTo.append((piecePos[0] - 1, piecePos[1] + 1))

                    # Generate actual moves
                    transformList = ['Q', 'R', 'N', 'B'] if change else ['P']
                    for move in allPossibleMovesTo:
                        ni, nj = move

                        for tp in transformList:
                            # Board
                            tempBoard = [row[:] for row in board]
                            tempBoard[ni][nj] = (color + tp)
                            tempBoard[piecePos[0]][piecePos[1]] = '_'

                            # Board List
                            tempBoardList = copy.deepcopy(boardList)
                            tempBoardList[color][piece].remove((piecePos[0], piecePos[1]))
                            tempBoardList[color][tp].append((ni, nj))

                            if ni - piecePos[0] != 0 and nj - piecePos[1] != 0:  # remove other piece
                                if board[ni][nj] != '_':
                                    tempBoardList[board[ni][nj][0]][board[ni][nj][1]].remove((ni, nj))
                                else:
                                    tempBoardList[board[ni + 1][nj][0]][board[ni + 1][nj][1]].remove((ni + 1, nj))
                                    tempBoard[ni + 1][nj] = '_'

                            new_position = Position(tempBoard, 'W',
                                                    (piece, piecePos[0], piecePos[1], ni, nj))
                            new_position.boardList = tempBoardList
                            new_position.enPassFrom = tempEnPassFrom
                            new_position.enPassTo = tempEnPassTo
                            new_position.score = position_score(new_position)

                            result.append(Node(new_position))

    return result


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

        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', 'WR', '_', '_', '_', '_', '_'],
        # ['WP', 'BP', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_']

    ]

    start_pos = Position(board, 'W')
    return start_pos


if __name__ == "__main__":
    depth = 5
    start_pos = generate_starting_position()

    # start_pos.enPassFrom = [(3, 1)]
    # start_pos.enPassTo = (2, 0)

    # print(coords_to_square(4, 5))

    next = generate_next_possible_positions(start_pos)
    for node in next:
        node.position.printBoard()
        print()

    # start_time_2 = time.time()
    # print("Winner optimized: ", play(start_pos, depth))
    # end_time_2 = time.time()
    # print(f"Execution time: {end_time_2 - start_time_2:.2f}s")


