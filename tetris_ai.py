import numpy as np
import pickle
import random
import subprocess
from tetris_engine import tEngine  # Enable access to tetris engine

# Q-Learning AI to play Tetris via tetris_terminal
engine = tEngine()  # Set engine to instance of tEngine

# Hyperparameters
ALPHA = 0.1      # Learning rate
GAMMA = 0.99     # Discount factor
EPSILON = 0.1    # Exploration rate
EPISODES = 1000  # Number of games to play

# Q-table: state-action values
Q = {}

def get_state(board):
    """Convert board to a tuple for hashing."""
    return tuple(map(tuple, board))

def choose_action(state, actions):
    """Epsilon-greedy action selection."""
    if random.random() < EPSILON:
        return random.choice(actions)
    qvals = [Q.get((state, a), 0) for a in actions]
    max_q = max(qvals)
    best_actions = [a for a, q in zip(actions, qvals) if q == max_q]
    return random.choice(best_actions)

def get_possible_actions():
    """Return all possible actions."""
    # Example: ['left', 'right', 'rotate', 'drop']
    return ['left', 'right', 'rotate', 'drop']

def run_tetris_action(proc, action):
    """Send action to tetris_terminal and get new state, reward, done."""
    proc.stdin.write((action + '\n').encode())
    proc.stdin.flush()
    # Read output from tetris_terminal (implement protocol as needed)
    board = np.zeros((20, 10))  # Replace with actual board parsing
    reward = 0     # Reward
    done = engine.game_over     # Game Over flag
    return board, reward, done

def main():
    for episode in range(EPISODES):
        # Start tetris_terminal as a subprocess
        proc = subprocess.Popen(
            ['python3', 'tetris_terminal.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        board = np.zeros((20, 10))  # Initial state (replace with actual)
        state = get_state(board)
        done = False

        while not done:
            actions = get_possible_actions()
            action = choose_action(state, actions)
            next_board, reward, done = run_tetris_action(proc, action)
            next_state = get_state(next_board)
            # Q-learning update
            old_q = Q.get((state, action), 0)
            next_qs = [Q.get((next_state, a), 0) for a in actions]
            Q[(state, action)] = old_q + ALPHA * (
                reward + GAMMA * max(next_qs, default=0) - old_q
            )
            state = next_state

        proc.terminate()

    # Save Q-table
    with open('q-ai/tetris_q_table.pkl', 'wb') as f:
        pickle.dump(Q, f)

if __name__ == '__main__':
    main()