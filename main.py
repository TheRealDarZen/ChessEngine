import time


absolute_score = 1000

class Position:
    def __init__(self, board, move, last_move=None, score=0.0, movedKings=[], availRooks=[]):
        self.board = board
        self.move = move
        self.last_move = last_move
        self.winner = '_'
        self.score = score
        self.enPassFrom = []
        self.enPassTo = None
        self.movedKings = movedKings
        self.boardList = {
            'W': {
                'K': [],
                'Q': [],
                'R': [],
                'B': [],
                'N': [],
                'P': []
            },
            'B': {
                'K': [],
                'Q': [],
                'R': [],
                'B': [],
                'N': [],
                'P': []
            }
        }
        for i in range(8):
            for j in range(8):
                if board[i][j] == '_':
                    continue
                self.boardList[board[i][j][0]][board[i][j][1]].append((i, j))

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
    if i == -1 and j == -1:
        return ''
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


def is_under_attack(position, color, i, j):

    attacker_color = 'B' if color == 'W' else 'W'
    board = position.board

    # Pawns
    if attacker_color == 'W':
        if i < 7:
            if j > 0 and board[i + 1][j - 1] == 'WP':
                return True
            if j < 7 and board[i + 1][j + 1] == 'WP':
                return True
    else:
        if i > 0:
            if j > 0 and board[i - 1][j - 1] == 'BP':
                return True
            if j < 7 and board[i - 1][j + 1] == 'BP':
                return True

    # Knights
    knight_moves = [
        (1, 2), (2, 1), (-1, 2), (-2, 1),
        (1, -2), (2, -1), (-1, -2), (-2, -1)
    ]

    for di, dj in knight_moves:
        ni, nj = i + di, j + dj
        if 0 <= ni < 8 and 0 <= nj < 8:
            if board[ni][nj] == attacker_color + 'N':
                return True

    # King
    king_moves = [
        (0, 1), (0, -1), (1, 1), (1, 0),
        (1, -1), (-1, 1), (-1, 0), (-1, -1)
    ]

    for di, dj in king_moves:
        ni, nj = i + di, j + dj
        if 0 <= ni < 8 and 0 <= nj < 8:
            if board[ni][nj] == attacker_color + 'K':
                return True

    # Directional vectors
    rook_directions = [
        (0, 1),  # right
        (1, 0),  # down
        (0, -1),  # left
        (-1, 0)  # up
    ]

    bishop_directions = [
        (1, 1),  # down-right
        (1, -1),  # down-left
        (-1, -1),  # up-left
        (-1, 1)  # up-right
    ]

    # Horizontal, Vertical
    for di, dj in rook_directions:
        for step in range(1, 8):
            ni, nj = i + di * step, j + dj * step
            if 0 <= ni < 8 and 0 <= nj < 8:
                if board[ni][nj] == attacker_color + 'R' or board[ni][nj] == attacker_color + 'Q':
                    return True
                elif board[ni][nj] != '_':
                    break
            else:
                break

    # Diagonal
    for di, dj in bishop_directions:
        for step in range(1, 8):
            ni, nj = i + di * step, j + dj * step
            if 0 <= ni < 8 and 0 <= nj < 8:
                if board[ni][nj] == attacker_color + 'B' or board[ni][nj] == attacker_color + 'Q':
                    return True
                elif board[ni][nj] != '_':
                    break
            else:
                break

    return False


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
    board = position.board

    for i in range(8):
        for j in range(8):
            if board[i][j] == '_':
                continue
            piece = board[i][j][1]
            temp = pieces_score(piece)
            if piece != 'P':
                temp -= ((abs(i - 5) + abs(j - 5)) if position.move == 'W' else (abs(i - 4) + abs(j - 4))) * 0.01
            else:
                temp -= ((abs(i - 7)) if position.move == 'W' else (abs(i))) * 0.02
            if board[i][j][0] == 'W':
                score += temp
            else:
                score -= temp

    return score


def minimax_with_tree_generation(position, depth, alpha=float('-inf'), beta=float('inf'),
                                 isRoot=False):

    global absolute_score

    #print("Depth: ", depth)

    score = position.score

    color = position.move
    is_maximizing = (color == 'W')

    next_positions = generate_next_possible_positions(position)

    # Mate / Stalemate
    if not next_positions:
        ti, tj = position.boardList[color]['K'][0]
        if is_under_attack(position, color, ti, tj):
            score = absolute_score if color == 'W' else -absolute_score
        else:
            score = 0.0
        return None, score if isRoot else score

    if depth <= 0:
        return None, score if isRoot else score

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

                                if piece == 'K':
                                    if is_under_attack(position, color, ni, nj): # Attacked square
                                        break

                                # Board
                                tempBoard = [row[:] for row in board]
                                tempBoard[ni][nj] = (color + piece)
                                tempBoard[piecePos[0]][piecePos[1]] = '_'

                                new_position = Position(tempBoard, 'W' if color == 'B' else 'B', (piece, piecePos[0], piecePos[1], ni, nj, position.movedKings))
                                new_position.score = position_score(new_position)
                                if piece == 'K':
                                    new_position.movedKings.append(color)

                                try:
                                    ti, tj = new_position.boardList[color]['K'][0]
                                except:
                                    position.printBoard()
                                    new_position.printBoard()
                                if not is_under_attack(new_position, color, ti, tj):
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

                            new_position = Position(tempBoard, 'B',
                                                    ('P', piecePos[0], piecePos[1], ni, nj, position.movedKings))
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

                            new_position = Position(tempBoard, 'W',
                                                    ('P', piecePos[0], piecePos[1], ni, nj, position.movedKings))
                            new_position.enPassFrom = tempEnPassFrom
                            new_position.enPassTo = tempEnPassTo
                            new_position.score = position_score(new_position)

                            result.append(Node(new_position))

    # TODO Rooks for castle !!!
    # Castles
    if color not in position.movedKings:
        if color == 'W':
            # Short Castle
            if board[0][3] == 'WK' and board[0][0] == 'WR' and board[0][1] == '_' and board[0][2] == '_':
                if (not is_under_attack(position, color, 0, 1) and not is_under_attack(position, color, 0, 2)
                        and not is_under_attack(position, color,0, 3)):
                    # Board
                    tempBoard = [row[:] for row in board]
                    tempBoard[0][1] = 'WK'
                    tempBoard[0][2] = 'WR'
                    tempBoard[0][0] = '_'
                    tempBoard[0][3] = '_'

                    new_position = Position(tempBoard, 'W' if color == 'B' else 'B',
                                            ('O-O', -1, -1, -1, -1, position.movedKings))
                    new_position.movedKings.append(color)
                    new_position.score = position_score(new_position)

                    result.append(Node(new_position))

            # Long Castle
            if board[0][3] == 'WK' and board[0][7] == 'WR' and board[0][4] == '_' and board[0][5] == '_' and board[0][6] == '_':
                if not is_under_attack(position, color,0, 3) and not is_under_attack(position, color,0, 4) and not is_under_attack(position, color,0, 5):
                    # Board
                    tempBoard = [row[:] for row in board]
                    tempBoard[0][5] = 'WK'
                    tempBoard[0][4] = 'WR'
                    tempBoard[0][0] = '_'
                    tempBoard[0][7] = '_'

                    new_position = Position(tempBoard, 'W' if color == 'B' else 'B',
                                            ('O-O-O', -1, -1, -1, -1, position.movedKings))
                    new_position.movedKings.append(color)
                    new_position.score = position_score(new_position)

                    result.append(Node(new_position))

        else:
            # Short Castle
            if board[7][3] == 'BK' and board[7][0] == 'BR' and board[7][1] == '_' and board[7][2] == '_':
                if not is_under_attack(position, color, 7, 1) and not is_under_attack(position, color,7, 2) and not is_under_attack(
                        position, color,7, 3):
                    # Board
                    tempBoard = [row[:] for row in board]
                    tempBoard[7][1] = 'BK'
                    tempBoard[7][2] = 'BR'
                    tempBoard[7][0] = '_'
                    tempBoard[7][3] = '_'

                    new_position = Position(tempBoard, 'W' if color == 'B' else 'B',
                                            ('O-O', -1, -1, -1, -1, position.movedKings))
                    new_position.movedKings.append(color)
                    new_position.score = position_score(new_position)

                    result.append(Node(new_position))

            # Long Castle
            if board[7][3] == 'BK' and board[7][7] == 'BR' and board[7][4] == '_' and board[7][5] == '_' and board[7][
                6] == '_':
                if not is_under_attack(position, color,7, 3) and not is_under_attack(position, color,7, 4) and not is_under_attack(
                        position, color,7, 5):
                    # Board
                    tempBoard = [row[:] for row in board]
                    tempBoard[7][5] = 'BK'
                    tempBoard[7][4] = 'BR'
                    tempBoard[7][0] = '_'
                    tempBoard[7][7] = '_'

                    new_position = Position(tempBoard, 'W' if color == 'B' else 'B',
                                            ('O-O-O', -1, -1, -1, -1, position.movedKings))
                    new_position.score = position_score(new_position)

                    result.append(Node(new_position))


    return result


def play(position, depth):

    if position.last_move:
        piece, lf, nf, lt, nt = position.last_move
        print((piece if piece != 'P' else '') + coords_to_square(lf, nf) + '-' + coords_to_square(lt, nt))

    best_move, _ = minimax_with_tree_generation(position, depth,
                                                float('-inf'), float('inf'),
                                                True)

    if not(best_move):
        print('No move')
        return

    return play(best_move.position, depth)


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

        # ['WR', '_', '_', 'WK', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', 'BQ', '_', '_', 'BK']

    ]

    start_pos = Position(board, 'W')
    return start_pos


if __name__ == "__main__":
    depth = 5
    start_pos = generate_starting_position()

    # start_pos.enPassFrom = [(3, 1)]
    # start_pos.enPassTo = (2, 0)

    # print(coords_to_square(4, 5))

    # start_pos.printBoard()

    # next = generate_next_possible_positions(start_pos)
    # for node in next:
    #     node.position.printBoard()
    #     print()

    # start_time_2 = time.time()
    # print("Winner optimized: ", play(start_pos, depth))
    # end_time_2 = time.time()
    # print(f"Execution time: {end_time_2 - start_time_2:.2f}s")

    play(start_pos, depth)
