"""Script to watch agent play"""
import threading
import time
import tetris_terminal
import pickle
import curses
import queue
from tetris_engine import tEngine
from copy import deepcopy
from agent import Agent

def watch_agent_play(agent, model_path="model/model.h5"):
    """Watch the agent play Tetris."""
    # Initialize the Tetris engine
    engine = tEngine()
    
    # Load the model if provided
    if model_path:
        agent.model = agent.build_model()
        agent.model.load_weights(model_path)
    
    # Communication queues for thread-safe operations
    action_queue = queue.Queue()
    display_queue = queue.Queue()

    def game_loop():
        """Main game loop to run the Tetris game."""
        while not engine.game_over:
            state = agent.get_state()
            action = agent.act(state)
            action_queue.put(action)
            time.sleep(0.05)  # Control the speed of the game

    def display_loop(stdscr):
        """Display loop to render the game on the terminal."""
        curses.curs_set(0)
        tetris_terminal.init_colors()
        
        while not engine.game_over:
            if not display_queue.empty():
                action = display_queue.get()
                if action == 'quit':
                    break
            
            tetris_terminal.draw_board(stdscr, engine)
            stdscr.refresh()
            time.sleep(0.05)

    # Start the game and display loops in separate threads
    threading.Thread(target=game_loop).start()
    curses.wrapper(display_loop)