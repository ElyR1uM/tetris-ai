# Base engine of the tetris game with all logic handled
# Now with full commenting so anyone can get how it works
# Terminology: dx = delta x, dy = delta y, nx = new x, ny = new y

# Piece selection
import random
import copy
import json

# Initialise the board width and height
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
# Points awarded for clearing lines (At Level 1):
# 4 - Single line
# 10 - Double line
# 30 - Triple line
# 120 - Tetris (4 Lines)
# The score is then multiplied by the level prior to the line clear, so a single line at level 3 would be worth 12 points.

TETROMINO_SHAPES  = {
    # These are used with the official Tetris color scheme
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

# Engine to run Tetris
class tEngine: 
    # Equivalent of a constructor
    def __init__(self):
        self.board = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.game_over = False
        self.spawn_piece()
        self.level = 1  # Initial level
        self.tick_rate = 1 / self.level  # seconds between automatic piece drops
        self.total_cleared = 0  # Total lines cleared, used to increase the level
        # For Troubleshooting, output files are created here
        with open("out/out0.txt", "w") as f:
            f.truncate(0)  # Clear the file at the start of the game
            f.write("The game logs any time lines are cleared\n")
        with open("out/out1.txt", "w") as f:
            f.truncate(0)
            f.write("Game log for total lines cleared\n")
        with open("out/out2.txt", "w") as f:
            f.truncate(0)
            f.write("Level Log\n")

    def spawn_piece(self):
        # Picks a random piece type from the shapes list
        self.piece_type = random.choice(list(TETROMINO_SHAPES.keys()))
        # Copies the attributes (In this case the shape) of the selected piece type so the game knows how to display and hanle it's collisions
        self.piece = copy.deepcopy(TETROMINO_SHAPES[self.piece_type])
        # Centers the piece
        self.piece_x = BOARD_WIDTH // 2 - len(self.piece[0]) // 2
        # Spawns the piece at the top of the board
        self.piece_y = 0
        # If the piece collides immediately upon spawning, end the game
        if self.check_collision():
            self.game_over = True
            # To be done: Fully stop the game, Save scores.

    # Handle collisions (Walls or other pieces)
    def check_collision(self, dx=0, dy=0, rotated_piece=None):
        # Set the shape: Rotate if rotated, otherwise remain the same
        shape = rotated_piece if rotated_piece else self.piece
        # "Imagine" the hitbox of the piece every tick
        for y, row in enumerate(shape):
            # Enumerate through each possible field
            for x, cell in enumerate(row):
                # Checks if the cell is filled (1) in the piece shape
                if cell:
                    # nx, ny = "New" x and y
                    nx = self.piece_x + x + dx
                    ny = self.piece_y + y + dy
                    # If nx or ny are out of bounds, the piece collides with the board
                    if nx < 0 or nx >= BOARD_WIDTH or ny >= BOARD_HEIGHT:
                        # returns collision
                        return True
                    if ny >= 0 and self.board[ny][nx]:
                        return True
        return False
    
    def move(self, dx):
        if not self.check_collision(dx=dx):
            # dx is negative when the left arrow is pressed and positive when the right arrow is pressed
            self.piece_x += dx

    def rotate(self):
        # Reimagine the hitbox of the piece by taking the original shape and rotating it 90 deg clockwise
        rotated = [list(row) for row in zip(*self.piece[::-1])]
        # Can't rotatie if the new rotated piece would collide with the board
        if not self.check_collision(rotated_piece=rotated):
            self.piece = rotated

    # Places the piece and moves on to the next one
    def lock_piece(self):
        for y, row in enumerate(self.piece):
            for x, cell in enumerate(row):
                if cell:
                    self.board[self.piece_y + y][self.piece_x + x] = 1

    # Removes fully filled lines and shifts the board down
    def clear_lines(self):
        new_board = [row for row in self.board if any(cell == 0 for cell in row)]
        cleared = BOARD_HEIGHT - len(new_board) # Gives you the ability to clear multiple lines at once
        with open("out/out0.txt", "a") as f:
            if cleared > 0:
                f.write(f"Cleared {cleared} lines\n")
        oldtc = self.total_cleared
        self.total_cleared += cleared        
        with open("out/out1.txt", "a") as f:
            if self.total_cleared > oldtc:
                f.write(f"Total cleared lines: {self.total_cleared}\n")
        # Debugging line to see how many lines were cleared
        # Add new empty lines at the top of the board to prevent the board from shrinking
        for _ in range(cleared):
            new_board.insert(0, [0] * BOARD_WIDTH)
        self.board = new_board
        switcher = {
            1: 4,  # Single line
            2: 10, # Double line
            3: 30, # Triple line
            4: 120 # Tetris (4 Lines)
        }
        self.score += switcher.get(cleared, 0) * self.level  # Multiply the score by the level

    # Difficulty scaling
    def increase_level(self):
        # Increases the level every 10 lines cleared, with a maximum of Level 10
        if self.total_cleared >= 10 & self.level < 10:
            self.level += 1
            self.total_cleared = 0  # Reset the cleared lines counter
            self.tick_rate = 1 / self.level
            with open("out/out2.txt", "a") as f:
                f.write(f"Level increased: {self.level - 1} => {self.level}\n")

    def calculate_efficiency(self): # This is supposed to calculate the efficiency of every turn the player does and should be a good parameter to pass on to the AI to "Improve"
        return 0

    def drop(self):
        if not self.check_collision(dy=1):
            self.piece_y += 1
        else:
            self.lock_piece()
            self.clear_lines()
            self.spawn_piece()

    # Spacebar; Hard Drop, Teleports the piece to the bottom of the board
    def hard_drop(self):
        while not self.check_collision(dy=1):
            self.piece_y += 1
        self.lock_piece()
        self.clear_lines()
        self.spawn_piece()
