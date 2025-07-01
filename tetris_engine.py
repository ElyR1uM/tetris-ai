"""Base game Engine"""

# Piece selection
import random
import copy
import json
import datetime
import numpy as np

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
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the game state to all initial values."""
        # Game reset
        self.board = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.last_score = 0
        self.game_over = False
        self.spawn_piece()
        self.level = 1
        self.tick_rate = 1 / self.level  # seconds between automatic piece drops
        self.cleared = 0 # Lines cleared this turn
        self.total_cleared = 0  # Total lines cleared since last level
        self.holes = 0
        self.bumpiness = 0
        self.heights = 0

    ### Piece Movement Functions

    def move(self, dx):
        """Moves the current piece either left or right."""
        if not self.check_collision(dx=dx):
            # dx is negative when the left arrow is pressed and positive when the right arrow is pressed
            self.piece_x += dx

    def rotate(self):
        """Rotates current piece clockwise."""
        # Reimagine the hitbox of the piece by taking the original shape and rotating it 90 deg clockwise
        rotated = [list(row) for row in zip(*self.piece[::-1])]
        # Can't rotate if the new rotated piece would collide with the board
        if not self.check_collision(rotated_piece=rotated):
            self.piece = rotated

    ### Handling Pieces

    def spawn_piece(self): # called in drop()
        """Spawns a new piece when the previous one is locked in place."""
        # Picks a random piece type from the shapes list
        if not self.game_over:
            self.piece_type = random.choice(list(TETROMINO_SHAPES.keys()))
            # Copies the attributes (In this case the shape) of the selected piece type so the game knows how to display and hanle it's collisions
            self.piece = copy.deepcopy(TETROMINO_SHAPES[self.piece_type])
            # Spawns piece at center
            self.piece_x = BOARD_WIDTH // 2 - len(self.piece[0]) // 2
            # Spawns the piece at top
            self.piece_y = 0
        # If the piece collides immediately upon spawning, end the game
        if self.check_collision():
            self.game_over = True

    def check_collision(self, dx=0, dy=0, rotated_piece=None):
        """Checks for collisions with the board or other pieces."""
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
    

    def lock_piece(self): # called in drop()
        """Locks a piece in place."""
        for y, row in enumerate(self.piece):
            for x, cell in enumerate(row):
                if cell:
                    board_x = self.piece_x + x
                    board_y = self.piece_y + y
                    if 0 <= board_x < BOARD_WIDTH and 0 <= board_y < BOARD_HEIGHT:
                        # Store the piece type instead of just True
                        self.board[board_y][board_x] = self.piece_type # type: ignore
                    #self.board[self.piece_y + y][self.piece_x + x] = 1

    def drop(self): # called in step()
        """Drops current piece by 1. Called every tick."""
        if not self.check_collision(dy=1):
            self.piece_y += 1
        else:
            self.lock_piece()
            self.clear_lines()
            self.spawn_piece()

    def hard_drop(self): # called in step()
        """Teleports current piece to the bottom of the board or until it collides with other pieces."""
        while not self.check_collision(dy=1):
            self.piece_y += 1
        self.lock_piece()
        self.clear_lines()
        self.spawn_piece()

    ### Board updates and calculations

    def clear_lines(self): # called in drop()
        """Counts and clears filled lines."""
        self.last_score = self.score
        new_board = [row for row in self.board if any(cell == 0 for cell in row)]
        self.cleared = BOARD_HEIGHT - len(new_board) # Gives you the ability to clear multiple lines at once
        self.total_cleared += self.cleared
        # Debugging line to see how many lines were cleared
        # Add new empty lines at the top of the board to prevent the board from shrinking
        for _ in range(self.cleared):
            new_board.insert(0, [0] * BOARD_WIDTH)
        self.board = new_board
        switcher = {
            1: 4,  # Single line
            2: 10, # Double line
            3: 30, # Triple line
            4: 120 # Tetris (4 Lines)
        }
        self.score += switcher.get(self.cleared, 0) * self.level  # Multiply the score by the level

    def get_reward(self): # called in step()
        """Calculates the reward for the current turn with the formula:\n
        Lines_cleared^2 * Board_width + 1"""
        reward = 0
        if not self.game_over:
            reward += self.cleared ** 2 * BOARD_WIDTH + 1
        else:
            reward -= 5
        return reward
    
    def get_bumpiness_heights(self): # called in get_state()
        """Calculates the bumpiness and hewlights of each column."""
        bumpiness = 0
        column_heights = [0] * BOARD_WIDTH

        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                if self.board[y][x] != 0:
                    column_heights[x] = BOARD_HEIGHT - y
                    break
        for i in range(1, len(column_heights)):
            bumpiness += abs(column_heights[i] - column_heights[i - 1])
        return bumpiness, sum(column_heights)
    
    def get_holes(self): # called in get_state()
        """Counts every empty cell that 'has a roof'."""
        self.holes = 0
        for x in range(BOARD_WIDTH):
            found_block = False
            for y in range(BOARD_HEIGHT):
                if self.board[y][x] != 0:
                    found_block = True
                elif found_block and self.board[y][x] == 0:
                    self.holes += 1
        return self.holes
    
    def increase_level(self): # called in step()
        """Increases the level and thus the speed and difficulty of the game every 10 lines cleared."""
        # Increases the level every 10 lines cleared, with a maximum of Level 10
        if self.total_cleared >= 10 and self.level < 10:
            self.total_cleared = 0  # Reset the cleared lines counter
            self.level += 1
            self.tick_rate = 1 / self.level

    def get_state(self): # called in step()
        """Returns an array with the lines cleared this turn, holes countes, bumpiness and heights of each column."""
        self.bumpiness, self.heights = self.get_bumpiness_heights()
        self.get_holes()
        return np.array([self.cleared, self.holes, self.bumpiness, self.heights])

    def get_possible_states(self): # Only relevant for the AI
        """Returns a dictionary with all possible states for the current piece."""
        states = {}
        original_piece = copy.deepcopy(self.piece)
        original_board = copy.deepcopy(self.board)
        original_x = self.piece_x
        original_y = self.piece_y

        for rotation in range(4):
            # Rotate the piece to the current rotation
            if rotation > 0:
                self.piece = [list(row) for row in zip(*self.piece[::-1])]
            shape_width = len(self.piece[0])
            # minimum position the piece can be placed at
            min_x = 0
            # maximum postition the piece can be placed at
            max_x = BOARD_WIDTH - shape_width

            for x in range(min_x, max_x + 1):
                self.piece_x = x
                self.piece_y = 0
                # Drop the piece down until collision
                while not self.check_collision(dy=1):
                    self.piece_y += 1
                # Lock the piece in place
                self.lock_piece()
                # Get the state after locking
                state = self.get_state()
                # Store the state with key (rotation, x)
                states[(rotation, x)] = state
                # Restore the board for the next iteration
                self.board = copy.deepcopy(original_board)
        # Restore original piece and position
        self.piece = original_piece
        self.piece_x = original_x
        self.piece_y = original_y
        return states

    
    def step(self):
        """Performed once per tick"""
        self.drop()
        self.increase_level()
        self.get_state()

    def write_scores(self, name): # called in tetris_terminal.py
        """Writes a score to scores.json if user agrees."""
        # Write the score as an entry in a JSON array in scores.json
        score_form = {
            "score": self.score,
            "date": datetime.datetime.now().strftime("%d/%m/%Y"), # UTC timezone if the highscore is achieved through the codespace
            "name": name,
        }
        try:
            with open("out/scores.json", "r") as f:
                try:
                    scores = json.load(f)
                    if not isinstance(scores, list):
                        scores = []
                except json.JSONDecodeError:
                    scores = []
        except FileNotFoundError:
            scores = []
        scores.append(score_form)
        with open("out/scores.json", "w") as f:
            json.dump(scores, f, indent=4)
