"""AI to play game, training is handled in tetris_trainer.py"""
import queue
import random
import time
import threading
import curses
import pickle
import tetris_terminal
from copy import deepcopy
from tetris_engine import tEngine

class qAgent:
    """Agent to play Tetris"""
    def __init__(self, q_table_path='q-ai/q_table.pkl'):
        """Initialize the Tetris AI agent."""

        # Communication queues for thread-safe operations
        self.action_queue = queue.Queue()
        self.display_queue = queue.Queue()

        # Thread control
        self.running = threading.Event()
        self.running.set()
        self.game_over_event = threading.Event()

        # Game State
        self.engine_lock = threading.Lock()
        # Gives direct access to the engine.
        self.engine = tEngine()

        # AI Components
        self.q_table = self.load_q_table(q_table_path)
        # Defines the possible actions the Agent can take each step
        self.actions = ['left', 'right', 'drop', 'rotate']

        # Timing
        self.refresh_rate = 0.05 # = 20 FPS
    
    def load_q_table(self, path):
        """Loads the Q-table"""
        try:
            with open(path, 'rb') as f:
                q_table = pickle.load(f)
                self.q_table = q_table
        except Exception as e:
            # Fallback if q table doesn't exist
            print(f"Error loading Q-table: {e}")
            return {}
    
    def get_state(self):
        """Get the current state of the game."""
        # Prevents multiple threads from accessing the engine at once
        with self.engine_lock:
            # Converts the board and piece state to a tuple
            board_state = tuple(map(tuple, self.engine.board))
            piece_info = (
                self.engine.piece_type,
                self.engine.piece_x,
                self.engine.piece_y,
                tuple(map(tuple, self.engine.piece))
            )
            game_info = {
                'score': self.engine.score,
                'level': self.engine.level,
                'efficiency': self.engine.efficiency,
                'game_over': self.engine.game_over
            }
            # Returns a tuple of board and piece info + game info dictionary
            return (board_state, piece_info), game_info
    
    def execute_action(self, action):
        """Execute action in game"""
        with self.engine_lock:
            if self.engine.game_over:
                return False

            if action == 'left':
                self.engine.move(-1)
            elif action == 'right':
                self.engine.move(1)
            elif action == 'rotate':
                self.engine.rotate()
            elif action == 'drop': # Note: Hard drop
                self.engine.hard_drop()
            
            return True
        
    def thinking_thread(self):
        """Separate thread for decision making."""
        while self.running.is_set():
            # Get the current state of the game
            state, game_info = self.get_state()
            if not game_info['game_over']:
                # Chooses action based on stored qvals if exists
                if self.q_table:
                    qvals = [self.q_table.get((state, a), 0) for a in self.actions]
                    max_q = max(qvals) if qvals else 0
                    best_action = [a for a, q in zip(self.actions, qvals) if q == max_q]
                    action = random.choice(best_action)
                else:
                    # No Q-Table => Choose random actions
                    action = random.choice(self.actions)
                
                # Execute chosen action
                if self.execute_action(action):
                    # Queue action
                    try:
                        self.display_queue.put({
                            'type': 'action',
                            'action': action,
                        }, timeout=0.01)
                    except queue.Full:
                        # Fallback
                        pass
            else:
                # Notifies Diplay thread that the game is over
                self.game_over_event.set()
            time.sleep(0.1)  # Sleep to prevent excessive CPU usage

    def physics_thread(self):
        """Handle Game physics."""
        last_drop = time.time()
        while self.running.is_set():
            current_time = time.time()

            with self.engine_lock:
                # Calls natural drop according to current tick rate
                if (current_time - last_drop >= self.engine.tick_rate and not self.engine.game_over):
                    self.engine.drop()
            time.sleep(0.1)

    def display_thread(self, stdscr):
        """Thread to handle game displaying with curses"""
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(10)
        tetris_terminal.init_colors()

        last_refresh = time.time()
        current_action = "WAITING"

        while self.running.is_set():
            current_time = time.time()

            # Check for display updates called by other threads
            try:
                while True:
                    # Checks if any new action is in the queue
                    display_info = self.display_queue.get_nowait()
                    if display_info['type'] == 'action':
                        current_action = display_info['action'].upper()
            except queue.Empty:
                # If nothing is found, skip
                pass
        
            if current_time - last_refresh >= self.refresh_rate:
                try:
                    # Get state
                    state, game_info = self.get_state()

                    # Create a copy of the game Engine
                    with self.engine_lock:
                        display_engine = deepcopy(self.engine)
                    
                    tetris_terminal.draw_board(stdscr, display_engine)

                    # Show AI Info on the side of the screen
                    try: 
                        stdscr.addstr(len(display_engine.board) + 5, 0,
                                    f"Current Action: {current_action:<8}")
                        stdscr.addstr(0, 25, "Press 'q' to quit, 'r' to restart")
                    except curses.error:
                        pass

                    # Handle Game Overs
                    if game_info ['game_over']:
                        stdscr.addstr(len(display_engine.board) + 7, 0, "=== AI GAME OVER ===")
                        stdscr.addstr(len(display_engine.board) + 8, 0, f"Final Score: {game_info['score']}")
                        stdscr.addstr(len(display_engine.board) + 10, 0, f"Efficiency: {game_info['efficiency']:.1f}%")
                        stdscr.addstr(len(display_engine.board) + 11, 0, "Press 'r' to restart or 'q' to quit")
                    
                    stdscr.refresh()
                    last_refresh = current_time
                except curses.error:
                    pass

            # Read user Input: quit / restart
            key = stdscr.getch()
            if key == ord('q'):
                self.running.clear()
                break
            elif key == ord('r'):
                self.restart_game()

    def restart_game(self):
        """Restarts the game"""
        with self.engine_lock:
            self.engine = tEngine()
            self.move_count = 0
            self.game_over_event.clear()
            # Clear any non-empty queues
            while not self.display_queue.empty():
                try:
                    self.display_queue.get_nowait()
                except queue.Empty:
                    break

    def run_game(self):
        """Runs the game"""
        # Start AI and Physics threads
        ai_thread = threading.Thread(target=self.thinking_thread, daemon=True)
        physics_thread = threading.Thread(target=self.physics_thread, daemon=True)

        ai_thread.start()
        physics_thread.start()

        try:
            curses.wrapper(self.display_thread)
        except KeyboardInterrupt:
            print("\nGame interrupted by user")
        finally:
            self.running.clear()

            ai_thread.join(timeout=1.0)
            physics_thread.join(timeout=1.0)
        
    
def main():
    """Runs the AI"""
    agent = qAgent()
    agent.run_game()

if __name__ == "__main__":
    main()