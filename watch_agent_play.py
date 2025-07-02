"""Script to watch agent play Tetris visually"""
import curses
import time
import os
from tetris_engine import tEngine
from agent import Agent
import tetris_terminal

def watch_agent_play(stdscr, model_path="model/model.h5", delay=0.5):

    curses.curs_set(0)
    stdscr.nodelay(False)
    tetris_terminal.init_colors()
    
    # Check if model exists
    if not os.path.exists(model_path):
        stdscr.addstr(0, 0, f"Model not found at {model_path}")
        stdscr.addstr(1, 0, "Press any key to exit...")
        stdscr.refresh()
        stdscr.getch()
        return
    
    engine = tEngine()
    state_size = 10 * 20 + 4
    agent = Agent(state_size, model_path)
    
    # Prevents random actions
    agent.epsilon = 0.0
    
    stdscr.addstr(0, 0, "Loading model...")
    stdscr.refresh()
    time.sleep(1)
    
    game_count = 1
    
    while True:
        # Reset for new game
        engine.reset()
        
        stdscr.clear()
        stdscr.addstr(0, 0, f"Watching AI play - Game #{game_count}")
        stdscr.addstr(1, 0, f"Model: {model_path}")
        stdscr.addstr(2, 0, "Press 'q' to quit, 'r' to restart, 's' to skip to next game")
        stdscr.addstr(3, 0, "-" * 50)
        stdscr.refresh()
        time.sleep(0.05)
        
        # Game loop
        while not engine.game_over:
            # Check for user input
            stdscr.timeout(0)
            key = stdscr.getch()
            
            if key == ord('q'):
                return
            elif key == ord('r'):
                break  # Restart current game
            elif key == ord('s'):
                break  # Skip to next game
            
            # Get current state
            # current_state = engine.get_state()
            
            # Get all possible moves for current piece
            next_states = engine.get_possible_states()
            
            # If no moves possible, game is over
            if not next_states:
                break
            
            # Let agent choose the best action
            best_action = agent.act(next_states)
            
            if best_action is None:
                break
            
            rotation, x_pos = best_action
            
            # Apply the chosen action
            # Rotate the piece
            for _ in range(rotation):
                engine.rotate()
                # Show rotation animation
                tetris_terminal.draw_board(stdscr, engine)
                stdscr.addstr(0, 0, f"Game #{game_count} | Score: {engine.score} | Level: {engine.level}")
                stdscr.addstr(1, 0, f"Action: Rotate ({rotation} times), Move to x={x_pos}")
                stdscr.refresh()
                time.sleep(delay)
            
            # Move to target x position
            current_x = engine.piece_x
            if x_pos < current_x:
                # Move left
                for _ in range(current_x - x_pos):
                    engine.move(-1)
                    tetris_terminal.draw_board(stdscr, engine)
                    stdscr.addstr(0, 0, f"Game #{game_count} | Score: {engine.score} | Level: {engine.level}")
                    stdscr.addstr(1, 0, f"Action: Moving left to x={x_pos}")
                    stdscr.refresh()
                    time.sleep(delay)
            elif x_pos > current_x:
                # Move right
                for _ in range(x_pos - current_x):
                    engine.move(1)
                    tetris_terminal.draw_board(stdscr, engine)
                    stdscr.addstr(0, 0, f"Game #{game_count} | Score: {engine.score} | Level: {engine.level}")
                    stdscr.addstr(1, 0, f"Action: Moving right to x={x_pos}")
                    stdscr.refresh()
                    time.sleep(delay)
            
            # Hard drop piece
            engine.hard_drop()
            
            # Update display
            tetris_terminal.draw_board(stdscr, engine)
            stdscr.addstr(0, 0, f"Game #{game_count} | Score: {engine.score} | Level: {engine.level}")
            stdscr.addstr(1, 0, f"Action: Hard drop complete")
            if engine.cleared > 0:
                stdscr.addstr(2, 0, f"Lines cleared: {engine.cleared}")
            stdscr.refresh()
            # Pause between moves
            time.sleep(delay)
        
        # Game over screen
        if engine.game_over:
            tetris_terminal.draw_board(stdscr, engine)
            stdscr.addstr(0, 0, f"Game #{game_count} OVER | Final Score: {engine.score}")
            stdscr.addstr(len(engine.board) + 6, 0, "Game Over!")
            stdscr.addstr(len(engine.board) + 7, 0, f"Final Score: {engine.score}")
            stdscr.addstr(len(engine.board) + 8, 0, "Press 'n' for next game, 'q' to quit, 'r' to replay")
            stdscr.refresh()
            
            # Wait for user input
            stdscr.nodelay(False)
            while True:
                key = stdscr.getch()
                if key == ord('q'):
                    return
                elif key in [ord('n'), ord('r')]:
                    break
            
            if key == ord('n'):
                game_count += 1
        else:
            # Interrupt
            continue

def main():
    model_path = "model/model.h5"
    delay = 0.05
    
    try:
        curses.wrapper(watch_agent_play, model_path, delay)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()