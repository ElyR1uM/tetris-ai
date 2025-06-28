"""
Efficient DQN Trainer for Tetris AI using Convolutional Neural Networks
"""

import tensorflow as tf
import keras
from keras import models, layers
from keras.optimizers import Adam
import numpy as np
import pickle
import random
import json
import time
import os
from collections import deque, defaultdict
from copy import deepcopy

from tetris_engine import tEngine
from tetris_rewards import RewardCalculator

class DQNTetrisTrainer:
    def __init__(self, model_path='q-ai/dqn_model.h5', memory_size=50000):
        self.model_path = model_path
        self.stats_path = 'q-ai/dqn_training_stats.json'
        self.checkpoint_dir = 'q-ai/dqn_checkpoints'
        
        # Create directories
        os.makedirs('q-ai', exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # DQN hyperparameters
        self.learning_rate = 0.001
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.memory_size = memory_size
        self.target_update_freq = 1000  # steps
        
        # Game parameters
        self.board_height = 20
        self.board_width = 10
        self.actions = ['left', 'right', 'down', 'rotate', 'drop']
        self.num_actions = len(self.actions)
        
        # Cell type mapping - convert tetromino letters to numbers
        self.cell_mapping = {
            0: 0.0,      # Empty cell
            '0': 0.0,    # Empty cell (string)
            ' ': 0.0,    # Empty space
            'I': 1.0,    # I-piece
            'O': 2.0,    # O-piece
            'T': 3.0,    # T-piece
            'S': 4.0,    # S-piece
            'Z': 5.0,    # Z-piece
            'J': 6.0,    # J-piece
            'L': 7.0,    # L-piece
        }
        
        # Terminal handling for continuous training
        self.original_terminal_settings = None
        
        # Training components
        self.memory = deque(maxlen=memory_size)
        self.reward_calculator = RewardCalculator()
        
        # Neural networks
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.update_target_network()
        
        # Load existing model if available
        self.load_model()
        
        # Training statistics
        self.training_stats = self.load_stats()
        self.episode_count = 0
        self.step_count = 0
        
    def build_model(self):
        """Build CNN-based DQN model for Tetris board state"""
        model = models.Sequential([
            # Input: (batch_size, 20, 10, 1) - board state
            layers.Input(shape=(self.board_height, self.board_width, 1)),
            
            # Convolutional layers to detect patterns
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            
            # Global average pooling to reduce parameters
            layers.GlobalAveragePooling2D(),
            
            # Dense layers for decision making
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer - Q-values for each action
            layers.Dense(self.num_actions, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), # type: ignore
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def load_model(self):
        """Load existing model if available"""
        try:
            if os.path.exists(self.model_path):
                self.q_network = keras.models.load_model(self.model_path)
                self.target_network = keras.models.load_model(self.model_path)
                print(f"Loaded existing model from {self.model_path}")
            else:
                print("No existing model found, starting with fresh model")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with fresh model")
    
    def save_model(self):
        """Save the trained model"""
        try:
            self.q_network.save(self.model_path)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def preprocess_state(self, engine):
        """Convert game state to CNN input format"""
        try:
            # Create board representation with proper cell mapping
            board = np.zeros((self.board_height, self.board_width), dtype=np.float32)
            
            # Convert board cells using mapping dictionary
            for y in range(self.board_height):
                for x in range(self.board_width):
                    cell_value = engine.board[y][x]
                    board[y, x] = self.cell_mapping.get(cell_value, 0.0)
            
            # Add current piece to the board (temporary overlay)
            if hasattr(engine, 'piece') and engine.piece is not None:
                piece_y, piece_x = engine.piece_y, engine.piece_x
                piece = np.array(engine.piece)
                piece_type = getattr(engine, 'piece_type', 'T')
                piece_value = self.cell_mapping.get(piece_type, 8.0)  # Use 8.0 for current piece
                
                # Place piece on board (if it fits)
                for py in range(piece.shape[0]):
                    for px in range(piece.shape[1]):
                        if piece[py, px] != 0:
                            board_y, board_x = piece_y + py, piece_x + px
                            if (0 <= board_y < self.board_height and 
                                0 <= board_x < self.board_width):
                                board[board_y, board_x] = piece_value
            
            # Normalize values to [0, 1] range
            board = board / 8.0
            
            # Reshape for CNN: (height, width, channels)
            state = board.reshape(self.board_height, self.board_width, 1)
            
            return state
            
        except Exception as e:
            print(f"Error preprocessing state: {e}")
            # Return empty board as fallback
            return np.zeros((self.board_height, self.board_width, 1), dtype=np.float32)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Predict Q-values
        state_batch = np.expand_dims(state, axis=0)
        q_values = self.q_network.predict(state_batch, verbose=0)[0]
        return np.argmax(q_values)
    
    def execute_action(self, engine, action_idx):
        """Execute action on the engine"""
        action = self.actions[action_idx]
        try:
            if action == 'left':
                engine.move(-1)
            elif action == 'right':
                engine.move(1)
            elif action == 'down':
                engine.drop()
            elif action == 'rotate':
                engine.rotate()
            elif action == 'drop':
                engine.hard_drop()
            return True
        except Exception as e:
            print(f"Error executing action {action}: {e}")
            return False
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q-values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
    
    def train_single_episode(self, max_moves=2000, verbose=False):
        """Train AI for one episode"""
        engine = tEngine()
        total_reward = 0
        moves = 0
        
        current_state = self.preprocess_state(engine)
        episode_start = time.time()
        
        while not engine.game_over and moves < max_moves:
            # Choose and execute action
            action_idx = self.choose_action(current_state)
            
            # Store previous state for reward calculation
            prev_engine_state = deepcopy(engine)
            prev_full_state = self.get_full_state(engine)
            
            # Execute action
            if not self.execute_action(engine, action_idx):
                break
            
            # Get new state and calculate reward
            next_state = self.preprocess_state(engine)
            new_full_state = self.get_full_state(engine)
            
            reward = self.reward_calculator.calculate_reward(
                prev_full_state, self.actions[action_idx], new_full_state, engine
            )
            total_reward += reward
            
            # Store experience
            self.remember(
                current_state, action_idx, reward, next_state, engine.game_over
            )
            
            # Train the model
            if len(self.memory) > self.batch_size:
                self.replay()
            
            # Update target network periodically
            if self.step_count % self.target_update_freq == 0:
                self.update_target_network()
            
            current_state = next_state
            moves += 1
            self.step_count += 1
            
            if verbose and moves % 500 == 0:
                print(f"  Move {moves}, Score: {engine.score}, Reward: {total_reward:.1f}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        episode_time = time.time() - episode_start
        
        episode_stats = {
            'episode': self.episode_count,
            'score': engine.score,
            'moves': moves,
            'total_reward': total_reward,
            'lines_cleared': engine.total_cleared,
            'level': engine.level,
            'time': episode_time,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }
        
        return episode_stats
    
    def get_full_state(self, engine):
        """Get full state for reward calculation"""
        board_state = tuple(tuple(row) for row in engine.board)
        piece_info = (
            str(engine.piece_type),
            int(engine.piece_x),
            int(engine.piece_y),
            tuple(tuple(row) for row in engine.piece) if engine.piece is not None else ()
        )
        return (board_state, piece_info)
    
    def train_batch(self, num_episodes, save_interval=50, verbose=True):
        """Train for multiple episodes"""
        print(f"Starting DQN training for {num_episodes} episodes...")
        print(f"Model architecture: {self.q_network.count_params()} parameters")
        
        batch_start = time.time()
        episode_scores = []
        
        for episode in range(num_episodes):
            self.episode_count += 1
            
            try:
                episode_stats = self.train_single_episode(verbose=verbose and episode % 50 == 0)
                episode_scores.append(episode_stats['score'])
                
                # Save training stats
                if not hasattr(self, 'training_stats'):
                    self.training_stats = {'episodes': []}
                self.training_stats['episodes'].append(episode_stats)
                
                if verbose and (episode + 1) % 10 == 0:
                    recent_scores = episode_scores[-10:]
                    avg_score = sum(recent_scores) / len(recent_scores)
                    print(f"Episode {episode + 1}/{num_episodes}: "
                          f"Score={episode_stats['score']}, "
                          f"Avg={avg_score:.1f}, "
                          f"Lines={episode_stats['lines_cleared']}, "
                          f"ε={self.epsilon:.3f}")
                
                if (episode + 1) % save_interval == 0:
                    self.save_model()
                    self.save_stats()
                    print(f"Checkpoint saved at episode {episode + 1}")
                    
            except Exception as e:
                print(f"Error in episode {episode + 1}: {e}")
                continue
        
        batch_time = time.time() - batch_start
        print(f"\nTraining completed in {batch_time:.1f} seconds")
        print(f"Average score (last 100): {np.mean(episode_scores[-100:]):.1f}")
        print(f"Best score: {max(episode_scores)}")
        
        self.save_model()
        self.save_stats()
    
    def safe_print(self, message, end='\n'):
        """Print with proper formatting for both normal and raw terminal modes"""
        try:
            import select
            import termios
            import tty
            UNIX_TERMINAL = True
        except ImportError:
            UNIX_TERMINAL = False
        
        if UNIX_TERMINAL and self.original_terminal_settings is not None:
            # In raw mode, use explicit carriage return + newline
            if end == '\n':
                print(f"\r{message}\r\n", end='', flush=True)
            else:
                print(f"\r{message}", end=end, flush=True)
        else:
            # Normal printing
            print(message, end=end, flush=True)
    
    def train_continuous(self, save_interval=100, verbose=True):
        """Continuous training with interactive controls"""
        self.safe_print("Starting continuous DQN training...")
        self.safe_print("Commands during training:")
        self.safe_print("  's' - Save current progress")
        self.safe_print("  'q' - Quit training")
        self.safe_print("  'e' - Evaluate current performance")
        self.safe_print("  'p' - Pause/Resume training")
        self.safe_print("  '+' - Decrease save frequency (save more often)")
        self.safe_print("  '-' - Increase save frequency (save less often)")
        
        # Check for Unix terminal capabilities
        try:
            import select
            import termios
            import tty
            import sys
            UNIX_TERMINAL = True
        except ImportError:
            UNIX_TERMINAL = False
            self.safe_print("Note: Interactive commands not available on this platform")
            self.safe_print("Training will run continuously. Press Ctrl+C to stop.")
            input("Press Enter to start...")
            
            # Fall back to simple continuous training
            try:
                while True:
                    self.episode_count += 1
                    episode_stats = self.train_single_episode(verbose=False)
                    
                    if verbose and self.episode_count % 10 == 0:
                        print(f"Ep {self.episode_count}: Score={episode_stats['score']}, "
                              f"Lines={episode_stats['lines_cleared']}, ε={self.epsilon:.3f}")
                    
                    if self.episode_count % save_interval == 0:
                        self.save_model()
                        self.save_stats()
                        print(f"Auto-saved at episode {self.episode_count}")
                        
            except KeyboardInterrupt:
                self.safe_print(f"\nTraining stopped at episode {self.episode_count}")
            return

        self.safe_print("Press Enter to start...")
        input()

        # Set up terminal for non-blocking input
        try:
            self.original_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        except Exception as e:
            self.safe_print(f"Warning: Could not set up interactive mode: {e}")
            self.original_terminal_settings = None
            return

        batch_start = time.time()
        episodes_since_save = 0
        paused = False
        last_progress_line = ""

        try:
            while True:
                # Check for user input (non-blocking)
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    command = sys.stdin.read(1).lower()

                    # Clear the current progress line before showing command output
                    if last_progress_line:
                        self.safe_print(" " * len(last_progress_line), end='\r')
                    
                    if command == 'q':
                        self.safe_print("Quitting training...")
                        break
                    elif command == 's':
                        self.safe_print(f"Manual save at episode {self.episode_count}")
                        self.save_model()
                        self.save_stats()
                    elif command == 'e':
                        self.safe_print(f"Evaluating performance at episode {self.episode_count}")
                        self.evaluate_performance(5)
                    elif command == 'p':
                        paused = not paused
                        status = 'paused' if paused else 'resumed'
                        self.safe_print(f"Training {status}")
                    elif command == '+':
                        save_interval = max(10, save_interval - 50)
                        self.safe_print(f"Save interval decreased to {save_interval}")
                    elif command == '-':
                        save_interval += 50
                        self.safe_print(f"Save interval increased to {save_interval}")

                if not paused:
                    self.episode_count += 1
                    episodes_since_save += 1

                    episode_stats = self.train_single_episode(verbose=False)

                    # Show progress on same line
                    if verbose and self.episode_count % 5 == 0:
                        elapsed = time.time() - batch_start
                        
                        progress_line = (f"Ep {self.episode_count}: "
                                       f"Score={episode_stats['score']}, "
                                       f"Lines={episode_stats['lines_cleared']}, "
                                       f"ε={self.epsilon:.3f}, "
                                       f"Mem={len(self.memory)}, "
                                       f"Time={elapsed/60:.1f}m")
                        
                        # Clear previous line and print new progress
                        if last_progress_line:
                            self.safe_print(" " * len(last_progress_line), end='\r')
                        self.safe_print(progress_line, end='\r')
                        last_progress_line = progress_line

                    # Auto-save checkpoints
                    if episodes_since_save >= save_interval:
                        if last_progress_line:
                            self.safe_print(" " * len(last_progress_line), end='\r')
                        self.safe_print(f"Auto-saving at episode {self.episode_count}")
                        self.save_model()
                        self.save_stats()
                        episodes_since_save = 0
                        last_progress_line = ""
                else:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            if last_progress_line:
                self.safe_print(" " * len(last_progress_line), end='\r')
            self.safe_print(f"Training interrupted at episode {self.episode_count}")

        finally:
            # Always restore terminal settings
            if self.original_terminal_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_terminal_settings)
                    self.original_terminal_settings = None
                except Exception as e:
                    print(f"Warning: Could not restore terminal settings: {e}")
    
    
    def evaluate_performance(self, num_games=10):
        """Evaluate current model performance"""
        print(f"Evaluating model performance over {num_games} games...")
        
        old_epsilon = self.epsilon
        self.epsilon = 0  # No exploration during evaluation
        
        scores = []
        lines_cleared = []
        
        for game in range(num_games):
            stats = self.train_single_episode(max_moves=3000, verbose=False)
            scores.append(stats['score'])
            lines_cleared.append(stats['lines_cleared'])
            
            if (game + 1) % 5 == 0:
                print(f"  Game {game + 1}/{num_games}: Score={stats['score']}, Lines={stats['lines_cleared']}")
        
        self.epsilon = old_epsilon
        
        results = {
            'avg_score': np.mean(scores),
            'best_score': max(scores),
            'avg_lines': np.mean(lines_cleared),
            'best_lines': max(lines_cleared),
            'scores': scores
        }
        
        print("Evaluation Results:")
        print(f"Average Score: {results['avg_score']:.1f}")
        print(f"Best Score: {results['best_score']}")
        print(f"Average Lines: {results['avg_lines']:.1f}")
        print(f"Best Lines: {results['best_lines']}")
        
        return results
    
    def load_stats(self):
        """Load training statistics"""
        try:
            with open(self.stats_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'episodes': [], 'evaluations': []}
        except Exception as e:
            print(f"Error loading stats: {e}")
            return {'episodes': [], 'evaluations': []}
    
    def save_stats(self):
        """Save training statistics"""
        try:
            with open(self.stats_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
        except Exception as e:
            print(f"Error saving stats: {e}")

def main():
    """Main training interface"""
    trainer = DQNTetrisTrainer()
    
    print("DQN Tetris AI Trainer")
    print("====================")
    print("Commands:")
    print("  train <episodes>  - Train for specified episodes (default: 100)")
    print("  eval <games>      - Evaluate performance (default: 10)")
    print("  continuous        - Train continuously with interactive controls")
    print("  quick             - Quick training (50 episodes)")
    print("  long              - Long training (500 episodes)")
    print("  stats             - Show training statistics")
    print("  quit              - Exit trainer")
    print()
    
    while True:
        try:
            command = input("Enter command: ").strip().lower().split()
            
            if not command:
                continue
            
            if command[0] == 'quit':
                break
            
            elif command[0] == 'train':
                episodes = int(command[1]) if len(command) > 1 else 100
                trainer.train_batch(episodes)
            
            elif command[0] == 'eval':
                games = int(command[1]) if len(command) > 1 else 10
                trainer.evaluate_performance(games)
            
            elif command[0] == 'continuous':
                trainer.train_continuous()
            
            elif command[0] == 'quick':
                trainer.train_batch(50)
            
            elif command[0] == 'long':
                trainer.train_batch(500)
            
            elif command[0] == 'stats':
                stats = trainer.training_stats
                if stats['episodes']:
                    recent_episodes = stats['episodes'][-10:]
                    recent_scores = [ep['score'] for ep in recent_episodes]
                    print(f"Total episodes: {len(stats['episodes'])}")
                    print(f"Recent average score: {np.mean(recent_scores):.1f}")
                    print(f"Best score: {max(ep['score'] for ep in stats['episodes'])}")
                    print(f"Current epsilon: {trainer.epsilon:.4f}")
                    print(f"Memory usage: {len(trainer.memory)}/{trainer.memory_size}")
                else:
                    print("No training data available")
            
            else:
                print("Unknown command")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except ValueError:
            print("Invalid number format")
        except Exception as e:
            print(f"Error: {e}")
    
    print("Trainer finished!")

if __name__ == "__main__":
    main()