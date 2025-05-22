# Base engine of the tetris game with all logic handled

# Piece selection
import random
import copy

# Initialise the board width and height
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

TETROMINO_SHAPES  = {
    # These are to be used with the TINT color scheme because I think it looks really cool
    # Light Blue
    "I": [[1, 1, 1, 1]],
    # Yellow
    "O": [[1, 1],
          [1, 1]],
    # Magenta
    "T": [[0, 1, 0],
          [1, 1, 1]],
    # Green
    "S": [[0, 1, 1],
          [1, 1, 0]],
    # Red
    "Z": [[1, 1, 0],
          [0, 1, 1]],
    # Dark Blue
    "J": [[1, 0, 0],
          [1, 1, 1]],
    # Orange
    "L": [[0, 0, 1],
          [1, 1, 1]]
}

# Indian magic
class tEngine: 
    def __init__(self):
        self.board = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.game_over = False
        self.spawn_piece()

    def spawn_piece(self):
        self.piece_type = random.choice(list(TETROMINO_SHAPES.keys()))
        self.piece = copy.deepcopy(TETROMINO_SHAPES[self.piece_type])
        self.piece_x = BOARD_WIDTH // 2 - len(self.piece[0]) // 2
        self.piece_y = 0
        if self.check_collision():
            print("Game Over")
            self.game_over = True

    def check_collision(self, dx=0, dy=0, rotated_piece=None):
        shape = rotated_piece if rotated_piece else self.piece
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    nx = self.piece_x + x + dx
                    ny = self.piece_y + y + dy
                    if nx < 0 or nx >= BOARD_WIDTH or ny >= BOARD_HEIGHT:
                        return True
                    if ny >= 0 and self.board[ny][nx]:
                        return True
        return False
    
    def move(self, dx):
        if not self.check_collision(dx=dx):
            self.piece_x += dx

    def rotate(self):
        rotated = [list(row) for row in zip(*self.piece[::-1])]
        if not self.check_collision(rotated_piece=rotated):
            self.piece = rotated

    def lock_piece(self):
        print("Piece Locked")
        for y, row in enumerate(self.piece):
            for x, cell in enumerate(row):
                if cell:
                    self.board[self.piece_y + y][self.piece_x + x] = 1

    def clear_lines(self):
        new_board = [row for row in self.board if any(cell == 0 for cell in row)]
        cleared = BOARD_HEIGHT - len(new_board)
        for _ in range(cleared):
            new_board.insert(0, [0] * BOARD_WIDTH)
        self.board = new_board
        self.score += cleared

    def drop(self):
        if not self.check_collision(dy=1):
            self.piece_y += 1
        else:
            self.lock_piece()
            self.clear_lines()
            self.spawn_piece()

    def hard_drop(self):
        while not self.check_collision(dy=1):
            self.piece_y += 1
        self.lock_piece()
        self.clear_lines()
        self.spawn_piece()
