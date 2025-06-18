# Visual interface in the terminal for tetris_engine.py
# Curses is the library used to display the game
import curses
import time
import tetris_engine
import os
from tetris_engine import tEngine

# Map piece types to colors
PIECE_COLORS = {
    'I': curses.COLOR_CYAN,
    'O': curses.COLOR_YELLOW,
    'T': curses.COLOR_MAGENTA,
    'S': curses.COLOR_GREEN,
    'Z': curses.COLOR_RED,
    'J': curses.COLOR_BLUE,
    'L': curses.COLOR_WHITE,  # substitude for orange
}

game_score = 0

def init_colors():
    curses.start_color()
    curses.use_default_colors() # Ensures the color scheme fits with the terminal's theme
    for i, (ptype, color) in enumerate(PIECE_COLORS.items(), start=1):
        curses.init_pair(i, color, -1)  # foreground color, default background

# Function to get the color pair for a specific cell in the board -> Which cell is a background cell and which one is a piece cell?
def get_color_pair(engine, x, y):
    # Draws anything outside of the board as empty
    if y < 0 or y >= len(engine.board) or x < 0 or x >= len(engine.board[0]):
        return 0
    # Check if a cell if a part of a piece
    for piece_y, row in enumerate(engine.piece):
        for piece_x, cell in enumerate(row):
            if cell:
                px = engine.piece_x + piece_x
                py = engine.piece_y + piece_y
                if px == x and py == y:
                    return curses.color_pair(list(PIECE_COLORS.keys()).index(engine.piece_type) + 1)
    if engine.board[y][x]:
        # Try to guess piece type from lock pattern (optional)
        return curses.color_pair(7)  # white for locked blocks
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
        elif key == curses.KEY_LEFT:
            engine.move(-1)
        elif key == curses.KEY_RIGHT:
            engine.move(1)
        elif key == curses.KEY_DOWN: # Soft drop
            engine.drop()
        elif key == curses.KEY_UP: # Rotate
            engine.rotate()
        elif key == ord(' '): # Hard drop
            while not engine.check_collision(dy=1):
                engine.piece_y += 1
            # This is the part where all the functions of tEngine are called
            engine.lock_piece()
            engine.clear_lines()
            engine.spawn_piece()
            engine.increase_level()

        if time.time() - last_drop > engine.tick_rate and not engine.game_over:
            engine.drop()
            last_drop = time.time()

        draw_board(stdscr, engine)

        if engine.game_over:
            draw_board(stdscr, engine)
            stdscr.nodelay(False)
            stdscr.addstr(len(engine.board) + 3, 0, "Game Over! Press any key to exit.")
            stdscr.addstr(len(engine.board) + 4, 0, f"Final Score: {engine.score}")
            stdscr.addstr(len(engine.board) + 5, 0, "Do you want to save your score? (y/n)")
            stdscr.refresh()

            while True:
                key = stdscr.getch()
                if key in [ord('y'), ord('Y')]:
                    # Here you would save the score to a file or database
                    # For now, we just print it
                    stdscr.addstr(len(engine.board) + 6, 0, "Enter your name: ")
                    stdscr.refresh()
                    curses.echo()
                    name = stdscr.getstr(len(engine.board) + 4, len("Enter your name: ") + 1, 20).decode('utf-8')
                    curses.noecho()

                    engine.write_scores(name)
                    break
                elif key in [ord('n'), ord('N')]:
                    break
                elif key != -1:
                    break

if __name__ == "__main__":
    curses.wrapper(main)
