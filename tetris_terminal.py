# Visual interface in the terminal for tetris_engine.py
import curses
import time
from tetris_engine import tEngine

TICK_RATE = 0.5  # seconds per drop

def draw_board(stdscr, engine):
    stdscr.clear()
    board = [row[:] for row in engine.board]

    # Overlay the falling piece
    for y, row in enumerate(engine.piece):
        for x, cell in enumerate(row):
            if cell:
                bx = engine.piece_x + x
                by = engine.piece_y + y
                if 0 <= by < len(board) and 0 <= bx < len(board[0]):
                    board[by][bx] = 2  # 2 = active piece

    # Draw board
    for y, row in enumerate(board):
        stdscr.addstr(y, 0, "|")
        for x, cell in enumerate(row):
            if cell == 0:
                stdscr.addstr("  ")
            elif cell == 1:
                stdscr.addstr("██")  # Locked block
            else:
                stdscr.addstr("[]")  # Active piece
        stdscr.addstr("|\n")

    # Draw bottom border
    stdscr.addstr(len(board), 0, "+" + "--" * len(board[0]) + "+")
    stdscr.addstr(len(board) + 2, 0, f"Score: {engine.score}")
    if engine.game_over:
        stdscr.addstr(len(board) + 3, 0, "GAME OVER. Press Q to quit.")

    stdscr.refresh()


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)  # screen refresh every 100 ms

    engine = tEngine()
    last_drop = time.time()

    while True:
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key == curses.KEY_LEFT:
            engine.move(-1)
        elif key == curses.KEY_RIGHT:
            engine.move(1)
        elif key == curses.KEY_DOWN:
            engine.drop()
        elif key == curses.KEY_UP:
            engine.rotate()
        elif key == ord(' '):  # hard drop
            while not engine.check_collision(dy=1):
                engine.piece_y += 1
            engine.lock_piece()
            engine.clear_lines()
            engine.spawn_piece()

        if time.time() - last_drop > TICK_RATE:
            engine.drop()
            last_drop = time.time()

        draw_board(stdscr, engine)

        if engine.game_over and key == ord('q'):
            break

curses.wrapper(main)
