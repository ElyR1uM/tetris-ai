"""Visual user-driven Interface for tetris_engine."""

import curses
import time
import tetris_engine
from tetris_engine import tEngine

# Map piece types to colors
PIECE_COLORS = {
    # We start at 1 because curses color pairs start at 1 and 0 counts as background
    1: curses.COLOR_CYAN,
    2: curses.COLOR_YELLOW,
    3: curses.COLOR_MAGENTA,
    4: curses.COLOR_GREEN,
    5: curses.COLOR_RED,
    6: curses.COLOR_BLUE,
    7: 208
}

PIECE_TYPE_TO_ID = {
    "I": 1,
    "O": 2,
    "T": 3,
    "S": 4,
    "Z": 5,
    "J": 6,
    "L": 7
}

game_score = 0

def init_colors():
    """Sets the color for each piece."""

    curses.start_color()
    curses.use_default_colors() # Ensures the color scheme fits with the terminal's theme
    for i, (piece_id, color) in enumerate(PIECE_COLORS.items(), start=1):
        curses.init_pair(i, color, -1)  # foreground color, default background


def get_color_pair(engine, x, y):
    """Differentiates between piece cells and background cells."""

    # Background drawn as empty space
    if y < 0 or y >= len(engine.board) or x < 0 or x >= len(engine.board[0]):
        return 0
    
    # For falling pieces
    for piece_y, row in enumerate(engine.piece):
        for piece_x, cell in enumerate(row):
            if cell:
                px = engine.piece_x + piece_x
                py = engine.piece_y + piece_y
                if px == x and py == y:
                    # Convert piece type string to ID, then get color pair
                    piece_id = PIECE_TYPE_TO_ID.get(engine.piece_type, 0)
                    return curses.color_pair(piece_id)
    # For locked pieces (these should now be numeric IDs)
    if engine.board[y][x]:
        piece_id = engine.board[y][x]
        return curses.color_pair(piece_id)
    
    return 0

def draw_board(stdscr, engine):
    stdscr.clear()
    board = [row[:] for row in engine.board]

    height, width = stdscr.getmaxyx()
    required_height = len(engine.board) + 5
    required_width = tetris_engine.BOARD_WIDTH * 2 + 4

    if height < required_height or width < required_width:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Terminal too small. Resize to at least {required_width}x{required_height}.")
        stdscr.refresh()
        return


    # Draw the board
    for y in range(len(board)):
        stdscr.addstr(y, 0, "|")  # Left wall
        for x in range(len(board[0])):
            color = get_color_pair(engine, x, y)
            cell = engine.board[y][x]
            is_piece = False
            for py, row in enumerate(engine.piece):
                for px, pcell in enumerate(row):
                    if pcell and engine.piece_x + px == x and engine.piece_y + py == y:
                        is_piece = True
            if cell or is_piece:
                stdscr.addstr(y, 1 + x * 2, "██", color)
            else:
                stdscr.addstr(y, 1 + x * 2, "  ")
        stdscr.addstr(y, 1 + tetris_engine.BOARD_WIDTH * 2, "|")  # Right wall


    stdscr.addstr(len(board), 0, "+" + "--" * len(board[0]) + "+")
    stdscr.addstr(len(board) + 2, 0, f"Score: {engine.score}")
    stdscr.addstr(len(board) + 3, 0, f"Level: {engine.level}")
    stdscr.addstr(len(board) + 4, 0, f"Holes: {engine.holes}")
    stdscr.addstr(len(board) + 5, 0, f"Bumpiness: {engine.bumpiness}")
    stdscr.addstr(len(board) + 6, 0, f"Height: {engine.heights}")
    stdscr.refresh()

def main(stdscr):
    # Initialize the curses screen
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    init_colors()
    engine = tEngine()
    last_drop = time.time()

    while True:
        key = stdscr.getch()

        # Handle input
        if key == ord('q'): # Quit
            break
        elif key in [curses.KEY_LEFT, ord('j')]:
            engine.move(-1)
        elif key in [curses.KEY_RIGHT, ord('l')]:
            engine.move(1)
        elif key == curses.KEY_DOWN: # Soft drop
            engine.drop()
        elif key in [curses.KEY_UP, ord('k')]: # Rotate
            engine.rotate()
        elif key == ord(' '): # Hard drop
            while not engine.check_collision(dy=1):
                engine.piece_y += 1
            engine.step()

        if time.time() - last_drop > engine.tick_rate and not engine.game_over:
            engine.drop()
            last_drop = time.time()

        draw_board(stdscr, engine)

        if engine.game_over:
            # Game Over Screen
            draw_board(stdscr, engine)
            stdscr.nodelay(False)
            stdscr.addstr(len(engine.board) + 5, 0, "Game Over!")
            stdscr.addstr(len(engine.board) + 6, 0, f"Final Score: {engine.score}")
            stdscr.addstr(len(engine.board) + 7, 0, "Do you want to save your score? (y/n)")
            stdscr.refresh()

            while True:
                key = stdscr.getch()
                if key in [ord('y'), ord('Y')]: # Gives the option to write a small y or a capitalised Y
                    stdscr.addstr(len(engine.board) + 8, 0, "Enter your name: ")
                    stdscr.refresh()
                    curses.echo()
                    name = stdscr.getstr(len(engine.board) + 8, len("Enter your name: ") + 1, 20).decode('utf-8')
                    curses.noecho()

                    engine.write_scores(name)
                    stdscr.addstr(len(engine.board) + 8, 0, "Press any button to exit.")
                    break
                elif key in [ord('n'), ord('N')]:
                    stdscr.addstr(len(engine.board) + 9, 0, "Score not saved. Press any button to exit.")
                    break
                elif key in [ord('q'), ord('Q')]:
                    break

            stdscr.refresh()
            stdscr.getch()

            # Exits game loop
            return

if __name__ == "__main__":
    curses.wrapper(main)
