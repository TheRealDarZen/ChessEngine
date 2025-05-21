import time


absolute_score = 1000

class Position:
    def __init__(self, board, move, last_move=None, movedKings=None, availRooks=None, kings_coords=None, transform=None, numOfMoves=0):
        if movedKings is None:
            movedKings = []
        if availRooks is None:
            availRooks = [(0, 0), (0, 7), (7, 0), (7, 7)]
        if kings_coords is None:
            kings_coords = {
                'W': (0, 3),
                'B': (7, 3)
            }
        self.board = board
        self.move = move
        self.last_move = last_move
        self.winner = '_'
        self.enPassFrom = []
        self.enPassTo = None
        self.movedKings = movedKings
        self.availRooks = availRooks
        self.kings_coords = kings_coords
        self.transform = transform
        self.score = 0.0
        self.numOfMoves = numOfMoves
        # self.boardList = {
        #     'W': {
        #         'K': [],
        #         'Q': [],
        #         'R': [],
        #         'B': [],
        #         'N': [],
        #         'P': []
        #     },
        #     'B': {
        #         'K': [],
        #         'Q': [],
        #         'R': [],
        #         'B': [],
        #         'N': [],
        #         'P': []
        #     }
        # }
        # for i in range(8):
        #     for j in range(8):
        #         if board[i][j] == '_':
        #             continue
        #         self.boardList[board[i][j][0]][board[i][j][1]].append((i, j))

    def __getitem__(self, item):
        return item

    def printBoard(self):
        for row in self.board:
           print(*row, sep=' ')


def coords_to_square(i, j, t):
    letters = ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']
    if i == -1 and j == -1:
        return ''
    if t:
        letters[j] + (str)(i + 1) + t
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
        if i > 1:
            if j > 0 and board[i - 1][j - 1] == 'WP':
                return True
            if j < 7 and board[i - 1][j + 1] == 'WP':
                return True
    else:
        if i < 6:
            if j > 0 and board[i + 1][j - 1] == 'BP':
                return True
            if j < 7 and board[i + 1][j + 1] == 'BP':
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


def count_pieces(boardList):

    white_pieces = {'K': len(boardList['W']['K']),
                    'Q': len(boardList['W']['Q']),
                    'R': len(boardList['W']['R']),
                    'B': len(boardList['W']['B']),
                    'N': len(boardList['W']['N']),
                    'P': len(boardList['W']['P'])}

    black_pieces = {'K': len(boardList['B']['K']),
                    'Q': len(boardList['B']['Q']),
                    'R': len(boardList['B']['R']),
                    'B': len(boardList['B']['B']),
                    'N': len(boardList['B']['N']),
                    'P': len(boardList['B']['P'])}

    return white_pieces, black_pieces


def find_doubled_pawns(boardList, board):

    white_doubled = 0
    black_doubled = 0

    # Group pawns by file
    white_files = {}
    black_files = {}

    for i, j in boardList['W']['P']:
        white_files[j] = white_files.get(j, 0) + 1

    for i, j in boardList['B']['P']:
        black_files[j] = black_files.get(j, 0) + 1

    # Count doubled pawns in each file
    for file_index, pawn_count in white_files.items():
        if pawn_count > 1:
            white_doubled += pawn_count - 1

    for file_index, pawn_count in black_files.items():
        if pawn_count > 1:
            black_doubled += pawn_count - 1

    return white_doubled, black_doubled


def find_blocked_pawns(boardList, board):

    white_blocked = 0
    black_blocked = 0

    # Check each white pawn
    for i, j in boardList['W']['P']:
        if board[i + 1][j] != '_':
            white_blocked += 1

    for i, j in boardList['B']['P']:
        if board[i - 1][j] != '_':
            black_blocked += 1

    return white_blocked, black_blocked


def find_isolated_pawns(boardList, board):

    white_pawn_files = [False] * 8
    black_pawn_files = [False] * 8

    for i, j in boardList['W']['P']:
        white_pawn_files[j] = True

    for i, j in boardList['B']['P']:
        black_pawn_files[j] = True

    white_isolated = 0
    black_isolated = 0

    for i, j in boardList['W']['P']:
        has_adjacent = False
        if j > 0 and white_pawn_files[j - 1]:
            has_adjacent = True
        if j < 7 and white_pawn_files[j + 1]:
            has_adjacent = True

        if not has_adjacent:
            white_isolated += 1

    for i, j in boardList['B']['P']:
        has_adjacent = False
        if j > 0 and black_pawn_files[j - 1]:
            has_adjacent = True
        if j < 7 and black_pawn_files[j + 1]:
            has_adjacent = True

        if not has_adjacent:
            black_isolated += 1

    return white_isolated, black_isolated


def find_mobility_score(position):
    return position.numOfMoves if position.move == 'B' else -position.numOfMoves


def find_center_score(position, board):

    white_center_score = 0
    black_center_score = 0

    center = [(3, 3), (3, 4), (4, 3), (4, 4)]

    for i, j in center:
        if board[i][j][0] == 'W':
            white_center_score += 1
        elif board[i][j][0] == 'B':
            black_center_score += 1

        if is_under_attack(position, 'B', i, j):
            white_center_score += 0.5
        if is_under_attack(position, 'W', i, j):
            black_center_score += 0.5

    return white_center_score, black_center_score


def find_development_score(boardList, board):
    white_development_score = 0
    black_development_score = 0

    pieces = ['Q', 'N', 'B']

    for piece in pieces:
        for i, j in boardList['W'][piece]:
            if i != 0:
                white_development_score += 1 if piece != 'Q' else 0.5

        for i, j in boardList['B'][piece]:
            if i != 7:
                black_development_score += 1 if piece != 'Q' else 0.5

    return white_development_score, black_development_score


def position_score(position):

    board = position.board
    boardList = {
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
            boardList[board[i][j][0]][board[i][j][1]].append((i, j))

    white_pieces, black_pieces = count_pieces(boardList)

    white_doubled, black_doubled = find_doubled_pawns(boardList, board)
    white_blocked, black_blocked = find_blocked_pawns(boardList, board)
    white_isolated, black_isolated = find_isolated_pawns(boardList, board)
    mobility_score = find_mobility_score(position)
    white_center_score, black_center_score = find_center_score(position, board)
    white_development_score, black_development_score = find_development_score(boardList, board)


    score = (
            200 * (white_pieces['K'] - black_pieces['K']) +
            9 * (white_pieces['Q'] - black_pieces['Q']) +
            5 * (white_pieces['R'] - black_pieces['R']) +
            3 * (white_pieces['B'] - black_pieces['B'] + white_pieces['N'] - black_pieces['N']) +
            1 * (white_pieces['P'] - black_pieces['P']) -
            0.5 * ((white_doubled - black_doubled) +
                   (white_blocked - black_blocked) +
                   (white_isolated - black_isolated)) +
            0.15 * (white_center_score - black_center_score) +
            0.25 * (white_development_score - black_development_score)
    )

    return score


def minimax_with_tree_generation(position, depth, alpha=float('-inf'), beta=float('inf'),
                                 isRoot=False):

    global absolute_score

    #print("Depth: ", depth)

    color = position.move
    is_maximizing = (color == 'W')

    next_positions = generate_next_possible_positions(position)
    position.numOfMoves = len(next_positions)

    # Mate / Stalemate
    if not next_positions:
        ti, tj = position.kings_coords[color]
        if is_under_attack(position, color, ti, tj):
            position.score = absolute_score if color == 'B' else -absolute_score
            position.winner = 'W' if color == 'B' else 'B'
        else:
            position.score = 0.0
            position.winner = 'D'
        return position, position.score if isRoot else position.score

    if depth <= 0:
        position.score = position_score(position)
        return position, position.score if isRoot else position.score

    best_move = None

    if is_maximizing:
        best_score = float('-inf')
        for next_pos in next_positions:
            _, score = minimax_with_tree_generation(next_pos, depth - 1, alpha, beta, False)
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
            _, score = minimax_with_tree_generation(next_pos, depth - 1, alpha, beta, False)
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
    # boardList = position.boardList
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

    for i in range(8):
        for j in range(8):
            if board[i][j] == '_' or board[i][j][0] != color:
                continue
            piece = board[i][j][1]
            piecePos = (i, j)

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

                                new_position = Position(tempBoard, 'W' if color == 'B' else 'B',
                                                        (piece, piecePos[0], piecePos[1], ni, nj),
                                                        position.movedKings.copy(), position.availRooks.copy(), position.kings_coords.copy())

                                if piece == 'K':
                                    new_position.kings_coords[color] = (ni, nj)
                                    if color not in new_position.movedKings:
                                        new_position.movedKings.append(color)
                                elif piece == 'R':
                                    if piecePos in position.availRooks:
                                        new_position.availRooks.remove(piecePos)

                                ti, tj = new_position.kings_coords[color]
                                if not is_under_attack(new_position, color, ti, tj):
                                    result.append(new_position)

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

                    if piecePos[0] == 6:
                        change = True

                    if board[piecePos[0] + 1][piecePos[1]] == '_':

                        allPossibleMovesTo.append((piecePos[0] + 1, piecePos[1]))

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
                                                    ('P', piecePos[0], piecePos[1], ni, nj),
                                                    position.movedKings.copy(), position.availRooks.copy(), position.kings_coords.copy(),
                                                    tp if tp != 'P' else None)
                            new_position.enPassFrom = tempEnPassFrom
                            new_position.enPassTo = tempEnPassTo

                            ti, tj = new_position.kings_coords[color]
                            if not is_under_attack(new_position, color, ti, tj):
                                result.append(new_position)

                # Black
                else:

                    if piecePos[0] == 1:
                        change = True

                    if board[piecePos[0] - 1][piecePos[1]] == '_':

                        allPossibleMovesTo.append((piecePos[0] - 1, piecePos[1]))

                        if piecePos[0] == 6:  # 2nd rank
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
                                                    ('P', piecePos[0], piecePos[1], ni, nj),
                                                    position.movedKings.copy(), position.availRooks.copy(), position.kings_coords.copy(),
                                                    tp if tp != 'P' else None)
                            new_position.enPassFrom = tempEnPassFrom
                            new_position.enPassTo = tempEnPassTo

                            ti, tj = new_position.kings_coords[color]
                            if not is_under_attack(new_position, color, ti, tj):
                                result.append(new_position)

    # Castles
    if color not in position.movedKings:
        if color == 'W':
            # Short Castle
            if board[0][3] == 'WK' and board[0][0] == 'WR' and board[0][1] == '_' and board[0][2] == '_' and (0, 0) in position.availRooks:
                if (not is_under_attack(position, color, 0, 1) and not is_under_attack(position, color, 0, 2)
                        and not is_under_attack(position, color,0, 3)):
                    # Board
                    tempBoard = [row[:] for row in board]
                    tempBoard[0][1] = 'WK'
                    tempBoard[0][2] = 'WR'
                    tempBoard[0][0] = '_'
                    tempBoard[0][3] = '_'

                    new_position = Position(tempBoard, 'W' if color == 'B' else 'B',
                                            ('O-O', -1, -1, -1, -1), position.movedKings.copy(), position.availRooks.copy(), position.kings_coords.copy())
                    new_position.movedKings.append(color)
                    new_position.availRooks.remove((0, 0))

                    result.append(new_position)

            # Long Castle
            if board[0][3] == 'WK' and board[0][7] == 'WR' and board[0][4] == '_' and board[0][5] == '_' and board[0][6] == '_' and (0, 7) in position.availRooks:
                if not is_under_attack(position, color,0, 3) and not is_under_attack(position, color,0, 4) and not is_under_attack(position, color,0, 5):
                    # Board
                    tempBoard = [row[:] for row in board]
                    tempBoard[0][5] = 'WK'
                    tempBoard[0][4] = 'WR'
                    tempBoard[0][3] = '_'
                    tempBoard[0][7] = '_'

                    new_position = Position(tempBoard, 'W' if color == 'B' else 'B',
                                            ('O-O-O', -1, -1, -1, -1), position.movedKings.copy(), position.availRooks.copy(), position.kings_coords.copy())
                    new_position.movedKings.append(color)
                    new_position.availRooks.remove((0, 7))

                    result.append(new_position)

        else:
            # Short Castle
            if board[7][3] == 'BK' and board[7][0] == 'BR' and board[7][1] == '_' and board[7][2] == '_' and (7, 0) in position.availRooks:
                if not is_under_attack(position, color, 7, 1) and not is_under_attack(position, color,7, 2) and not is_under_attack(
                        position, color,7, 3):
                    # Board
                    tempBoard = [row[:] for row in board]
                    tempBoard[7][1] = 'BK'
                    tempBoard[7][2] = 'BR'
                    tempBoard[7][0] = '_'
                    tempBoard[7][3] = '_'

                    new_position = Position(tempBoard, 'W' if color == 'B' else 'B',
                                            ('O-O', -1, -1, -1, -1), position.movedKings.copy(), position.availRooks.copy(), position.kings_coords.copy())
                    new_position.movedKings.append(color)
                    new_position.availRooks.remove((7, 0))

                    result.append(new_position)

            # Long Castle
            if board[7][3] == 'BK' and board[7][7] == 'BR' and board[7][4] == '_' and board[7][5] == '_' and board[7][
                6] == '_' and (7, 7) in position.availRooks:
                if not is_under_attack(position, color,7, 3) and not is_under_attack(position, color,7, 4) and not is_under_attack(
                        position, color,7, 5):
                    # Board
                    tempBoard = [row[:] for row in board]
                    tempBoard[7][5] = 'BK'
                    tempBoard[7][4] = 'BR'
                    tempBoard[7][3] = '_'
                    tempBoard[7][7] = '_'

                    new_position = Position(tempBoard, 'W' if color == 'B' else 'B',
                                            ('O-O-O', -1, -1, -1, -1), position.movedKings.copy(), position.availRooks.copy(), position.kings_coords.copy())
                    new_position.movedKings.append(color)
                    new_position.availRooks.remove((7, 7))

                    result.append(new_position)


    return result


def print_move(position):
    piece, lf, nf, lt, nt = position.last_move
    print(
        (piece if piece != 'P' else '') + coords_to_square(lf, nf, None) + ('-' if lf != -1 else '') + coords_to_square(
            lt, nt, position.transform))


def play(position, depth):

    if position.last_move:
        print_move(position)

    best_move, _ = minimax_with_tree_generation(position, depth,
                                                float('-inf'), float('inf'),
                                                True)

    if not(best_move):
        print('No move')
        return

    if best_move.winner == 'W':
        print_move(best_move)
        return 'White wins!'
    elif best_move.winner == 'B':
        print_move(best_move)
        return 'Black wins!'
    elif best_move.winner == 'D':
        print_move(best_move)
        return 'Draw!'

    return play(best_move, depth)


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

        # ['WK', '_', '_', '_', '_', '_', '_', 'WR'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['WR', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', 'BK', '_', '_', '_', '_'],

        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],
        # ['_', '_', '_', '_', '_', '_', '_', '_'],

    ]

    start_pos = Position(board, 'W')
    return start_pos


if __name__ == "__main__":
    depth = 4
    start_pos = generate_starting_position()

    # start_pos.enPassFrom = [(3, 1)]
    # start_pos.enPassTo = (2, 0)

    # print(coords_to_square(4, 5))

    # start_pos.printBoard()

    # next = generate_next_possible_positions(start_pos)
    # for pos in next:
    #     pos.printBoard()
    #     pos.numOfMoves = len(next)
    #     print(position_score(pos))
    #     print()

    # start_time_2 = time.time()
    # print("Winner optimized: ", play(start_pos, depth))
    # end_time_2 = time.time()
    # print(f"Execution time: {end_time_2 - start_time_2:.2f}s")

    print(play(start_pos, depth))
