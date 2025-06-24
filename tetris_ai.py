#!/usr/bin/env python3
"""
Multithreaded Tetris AI with proper thread safety
"""

import curses
import time
import pickle
import random
import os
import threading
import queue
from copy import deepcopy
from tetris_engine import tEngine
import tetris_terminal

class ThreadSafeTetrisAI:
    def __init__(self, q_table_path='q-ai/tetris_q_table.pkl'):
        # Thread-safe communication
        self.action_queue = queue.Queue()
        self.state_queue = queue.Queue()
        self.display_queue = queue.Queue()
        
        # Thread control
        self.running = threading.Event()
        self.running.set()
        self.game_over_event = threading.Event()
        
        # Game state (protected by locks)
        self.engine_lock = threading.Lock()
        self.engine = tEngine()
        
        # AI components
        self.q_table = self.load_q_table(q_table_path)
        self.actions = ['left', 'right', 'down', 'rotate', 'drop']
        self.move_count = 0
        
        # Timing controls
        self.ai_move_delay = 0.3
        self.display_refresh_rate = 0.05  # 20 FPS
        
    def load_q_table(self, filepath):
        """Load Q-table thread-safely"""
        try:
            with open(filepath, 'rb') as f:
                q_table = pickle.load(f)
                print(f"Loaded Q-table with {len(q_table)} entries")
                return q_table
        except FileNotFoundError:
            print(f"Q-table file {filepath} not found. AI will use random actions.")
            return {}
    
    def get_game_state(self):
        """Get thread-safe copy of game state"""
        with self.engine_lock:
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
            return (board_state, piece_info), game_info
    
    def execute_action_safely(self, action):
        """Execute action with thread safety"""
        with self.engine_lock:
            if self.engine.game_over:
                return False
                
            if action == 'left':
                self.engine.move(-1)
            elif action == 'right':
                self.engine.move(1)
            elif action == 'down':
                self.engine.drop()
            elif action == 'rotate':
                self.engine.rotate()
            elif action == 'drop':
                # Hard drop
                while not self.engine.check_collision(dy=1):
                    self.engine.piece_y += 1
                self.engine.lock_piece()
                self.engine.clear_lines()
                self.engine.spawn_piece()
                self.engine.increase_level()
            
            return True
    
    def ai_thinking_thread(self):
        """AI decision-making thread"""
        last_ai_move = time.time()
        
        while self.running.is_set():
            current_time = time.time()
            
            # Make AI decision
            if current_time - last_ai_move >= self.ai_move_delay:
                state, game_info = self.get_game_state()
                
                if not game_info['game_over']:
                    # Choose action
                    if self.q_table:
                        qvals = [self.q_table.get((state, a), 0) for a in self.actions]
                        max_q = max(qvals) if qvals else 0
                        best_actions = [a for a, q in zip(self.actions, qvals) if q == max_q]
                        action = random.choice(best_actions)
                    else:
                        action = random.choice(self.actions)
                    
                    # Execute action
                    if self.execute_action_safely(action):
                        self.move_count += 1
                        # Send action info to display thread
                        try:
                            self.display_queue.put({
                                'type': 'action',
                                'action': action,
                                'move_count': self.move_count
                            }, timeout=0.01)
                        except queue.Full:
                            pass
                    
                    last_ai_move = current_time
                else:
                    self.game_over_event.set()
            
            time.sleep(0.01)  # Small sleep to prevent excessive CPU usage
    
    def game_physics_thread(self):
        """Handle game physics (gravity, timing)"""
        last_drop = time.time()
        
        while self.running.is_set():
            current_time = time.time()
            
            with self.engine_lock:
                # Natural gravity drop
                if (current_time - last_drop >= self.engine.tick_rate and 
                    not self.engine.game_over):
                    self.engine.drop()
                    last_drop = current_time
            
            time.sleep(0.01)
    
    def display_thread_func(self, stdscr):
        """Display thread - handles all curses operations"""
        # Initialize curses (only in this thread)
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(10)
        tetris_terminal.init_colors()
        
        last_refresh = time.time()
        current_action = "WAITING"
        move_count = 0
        
        while self.running.is_set():
            current_time = time.time()
            
            # Check for display updates from other threads
            try:
                while True:
                    display_info = self.display_queue.get_nowait()
                    if display_info['type'] == 'action':
                        current_action = display_info['action'].upper()
                        move_count = display_info['move_count']
            except queue.Empty:
                pass
            
            # Refresh display at controlled rate
            if current_time - last_refresh >= self.display_refresh_rate:
                try:
                    # Get current game state
                    state, game_info = self.get_game_state()
                    
                    # Create a copy of engine for display (thread-safe)
                    with self.engine_lock:
                        display_engine = deepcopy(self.engine)
                    
                    # Draw the game
                    tetris_terminal.draw_board(stdscr, display_engine)
                    
                    # Add AI info
                    try:
                        stdscr.addstr(len(display_engine.board) + 5, 0, 
                                     f"AI Action: {current_action:<8} Move: {move_count}")
                        stdscr.addstr(len(display_engine.board) + 6, 0,
                                     f"Q-table size: {len(self.q_table)}")
                        stdscr.addstr(0, 25, "Press 'q' to quit, 'r' to restart")
                    except curses.error:
                        pass
                    
                    # Handle game over
                    if game_info['game_over']:
                        stdscr.addstr(len(display_engine.board) + 7, 0, "=== AI GAME OVER ===")
                        stdscr.addstr(len(display_engine.board) + 8, 0, f"Final Score: {game_info['score']}")
                        stdscr.addstr(len(display_engine.board) + 9, 0, f"Total Moves: {move_count}")
                        stdscr.addstr(len(display_engine.board) + 10, 0, f"Efficiency: {game_info['efficiency']:.1f}%")
                        stdscr.addstr(len(display_engine.board) + 11, 0, "Press 'r' to restart or 'q' to quit")
                    
                    stdscr.refresh()
                    last_refresh = current_time
                    
                except curses.error:
                    # Handle terminal resize gracefully
                    pass
            
            # Check for user input
            key = stdscr.getch()
            if key == ord('q'):
                self.running.clear()
                break
            elif key == ord('r'):
                self.restart_game()
            
            time.sleep(0.01)
    
    def restart_game(self):
        """Restart the game thread-safely"""
        with self.engine_lock:
            self.engine = tEngine()
            self.move_count = 0
            self.game_over_event.clear()
            # Clear queues
            while not self.display_queue.empty():
                try:
                    self.display_queue.get_nowait()
                except queue.Empty:
                    break
    
    def run_multithreaded_game(self):
        """Main entry point for multithreaded game"""
        # Start AI and physics threads
        ai_thread = threading.Thread(target=self.ai_thinking_thread, daemon=True)
        physics_thread = threading.Thread(target=self.game_physics_thread, daemon=True)
        
        ai_thread.start()
        physics_thread.start()
        
        try:
            # Run display in main thread (curses requirement)
            curses.wrapper(self.display_thread_func)
        except KeyboardInterrupt:
            print("\nGame interrupted by user")
        finally:
            self.running.clear()
            
            # Wait for threads to finish
            ai_thread.join(timeout=1.0)
            physics_thread.join(timeout=1.0)


def main():
    """Entry point"""
    print("Starting Multithreaded AI Tetris...")
    print("Features:")
    print("- AI decision making in separate thread")
    print("- Game physics in separate thread") 
    print("- Display updates in main thread")
    print("- Thread-safe state management")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 'r' to restart game")
    print()
    
    game = ThreadSafeTetrisAI()
    game.run_multithreaded_game()
    
    print("Game finished!")

if __name__ == "__main__":
    main()