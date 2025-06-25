# Base engine of the tetris game with all logic handled
# Now with full commenting so anyone can get how it works
# Terminology: dx = delta x, dy = delta y, nx = new x, ny = new y

# Piece selection
import random
import copy
import json
import datetime

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
        self.reset()
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
        with open("out/out3.txt", "w") as f:
            f.truncate(0)
            f.write("Floating cells log\n")

    def reset(self):
        # Game reset
        self.board = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.last_score = 0
        self.game_over = False
        self.spawn_piece()
        self.level = 1
        self.tick_rate = 1 / self.level  # seconds between automatic piece drops
        self.prevtc = 0
        self.total_cleared = 0  # Total lines cleared, used to increase the level
        self.floating_cells = 0  # Used to calculate efficiency
        self.efficiency = 0

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
            # To be done: Fully stop the game

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
        self.calculate_efficiency()

    # Removes fully filled lines and shifts the board down
    def clear_lines(self):
        self.last_score = self.score
        new_board = [row for row in self.board if any(cell == 0 for cell in row)]
        cleared = BOARD_HEIGHT - len(new_board) # Gives you the ability to clear multiple lines at once
        with open("out/out0.txt", "a") as f:
            if cleared > 0:
                f.write(f"Cleared {cleared} lines\n")
        self.prevtc = self.total_cleared
        self.total_cleared += cleared
        with open("out/out1.txt", "a") as f:
            if self.total_cleared > self.prevtc:
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
        # Right now this is bugged, increasing levels every 2 or so lines
        if self.total_cleared >= 10 and self.level < 10:
            self.total_cleared = 0  # Reset the cleared lines counter
            self.level += 1
            self.tick_rate = 1 / self.level
            with open("out/out2.txt", "a") as f:
                f.write(f"Level increased: {self.level - 1} => {self.level}\n")
    
    def calculate_efficiency(self, shape_matrix=None):
        """
        Calculate an efficiency metric based on:
        - Height-based scoring: Penalizes high columns
        - Hole counting: Heavily penalizes gaps beneath blocks
        - Line completion: Rewards near-complete rows
        - Bumpiness/smoothness: Penalizes height differences between columns
        - Well formation: Considers deep vertical gaps

        Returns efficiency score from 0-100 (higher is better)
        """
        if shape_matrix is None:
            shape_matrix = self.piece

        # Create a temporary board with the current piece placed
        temp_board = [row[:] for row in self.board]

        # Place the current piece on the temporary board
        for dy, row in enumerate(shape_matrix):
            for dx, cell in enumerate(row):
                if cell:
                    board_x = self.piece_x + dx
                    board_y = self.piece_y + dy
                    if 0 <= board_x < BOARD_WIDTH and 0 <= board_y < BOARD_HEIGHT:
                        temp_board[board_y][board_x] = 1

        # 1. HEIGHT-BASED SCORING
        column_heights = []
        for x in range(BOARD_WIDTH):
            height = 0
            for y in range(BOARD_HEIGHT):
                if temp_board[y][x] == 1:
                    height = BOARD_HEIGHT - y
                    break
            column_heights.append(height)

        max_height = max(column_heights) if column_heights else 0
        avg_height = sum(column_heights) / len(column_heights)

        # Height penalty (0-30 points, lower is better)
        height_score = max(0, 30 - (max_height * 1.5 + avg_height * 0.5))

        # 2. HOLE COUNTING
        holes = 0
        for x in range(BOARD_WIDTH):
            found_block = False
            for y in range(BOARD_HEIGHT):
                if temp_board[y][x] == 1:
                    found_block = True
                elif found_block and temp_board[y][x] == 0:
                    holes += 1

        # Hole penalty (0-25 points, fewer holes is better)
        hole_score = max(0, 25 - holes * 3)

        # 3. LINE COMPLETION POTENTIAL
        line_completion_score = 0
        for y in range(BOARD_HEIGHT):
            filled_cells = sum(temp_board[y])
            if filled_cells == BOARD_WIDTH:
                line_completion_score += 10  # Complete line bonus
            elif filled_cells >= BOARD_WIDTH - 2:
                line_completion_score += 5   # Nearly complete line bonus
            elif filled_cells >= BOARD_WIDTH - 3:
                line_completion_score += 2   # Moderately complete line bonus

        # Cap line completion score at 20 points
        line_completion_score = min(20, line_completion_score)

        # 4. BUMPINESS/SMOOTHNESS
        bumpiness = 0
        for i in range(len(column_heights) - 1):
            bumpiness += abs(column_heights[i] - column_heights[i + 1])

        # Smoothness score (0-15 points, less bumpiness is better)
        smoothness_score = max(0, 15 - bumpiness * 0.5)

        # 5. WELL FORMATION
        well_score = 0
        for x in range(BOARD_WIDTH):
            left_height = column_heights[x-1] if x > 0 else 0
            right_height = column_heights[x+1] if x < BOARD_WIDTH-1 else 0
            current_height = column_heights[x]

            # Check if this forms a well (significantly lower than neighbors)
            if (left_height - current_height >= 3 and right_height - current_height >= 3):
                well_depth = min(left_height - current_height, right_height - current_height)
                # Wells can be good for T-spins and line setups, but very deep wells are bad
                if well_depth <= 4:
                    well_score += well_depth * 1.5  # Moderate wells are good
                else:
                    well_score -= (well_depth - 4) * 2  # Very deep wells are bad

        # Cap well score between -10 and 10
        well_score = max(-10, min(10, well_score))

        # 6. FLOATING CELLS (original metric)
        floating_cells = 0
        total_piece_cells = 0

        for dy, row in enumerate(shape_matrix):
            for dx, cell in enumerate(row):
                if cell:
                    total_piece_cells += 1
                    board_x = self.piece_x + dx
                    board_y = self.piece_y + dy
                    # Check if below is empty (within the board)
                    if (board_y + 1 < BOARD_HEIGHT and 
                        0 <= board_x < BOARD_WIDTH and 
                        temp_board[board_y + 1][board_x] == 0):
                        floating_cells += 1

        # Floating penalty (0-10 points)
        if total_piece_cells > 0:
            floating_score = 10 * (total_piece_cells - floating_cells) / total_piece_cells
        else:
            floating_score = 10

        # COMBINE ALL SCORES
        total_score = (height_score +      # 30 points max
                       hole_score +        # 25 points max  
                       line_completion_score + # 20 points max
                       smoothness_score +  # 15 points max
                       floating_score +    # 10 points max
                       well_score)         # -10 to +10 points

        # Convert to percentage (total possible: 110 points)
        efficiency_percentage = round(min(100, max(0, (total_score / 110) * 100)))

        # Store individual components for debugging
        self.efficiency_breakdown = {
            'height_score': height_score,
            'hole_score': hole_score,
            'line_completion_score': line_completion_score,
            'smoothness_score': smoothness_score,
            'floating_score': floating_score,
            'well_score': well_score,
            'total_score': total_score,
            'max_height': max_height,
            'avg_height': avg_height,
            'holes': holes,
            'bumpiness': bumpiness,
            'floating_cells': floating_cells
        }

        self.efficiency = efficiency_percentage

        # Enhanced logging
        with open("out/out3.txt", "a") as f:
            f.write(f"=== Efficiency Breakdown ===\n")
            f.write(f"Height Score: {height_score:.1f}/30 (Max: {max_height}, Avg: {avg_height:.1f})\n")
            f.write(f"Hole Score: {hole_score:.1f}/25 (Holes: {holes})\n")
            f.write(f"Line Completion: {line_completion_score:.1f}/20\n")
            f.write(f"Smoothness: {smoothness_score:.1f}/15 (Bumpiness: {bumpiness:.1f})\n")
            f.write(f"Floating: {floating_score:.1f}/10 (Floating cells: {floating_cells})\n")
            f.write(f"Wells: {well_score:.1f} (-10 to +10)\n")
            f.write(f"Total Efficiency: {efficiency_percentage:.1f}%\n")
            f.write(f"Column Heights: {column_heights}\n\n")
    
        return efficiency_percentage

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

    def write_scores(self, name):
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
