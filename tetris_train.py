"""
Trainer Winkler
"""

import pickle
import json
import random
import time
import os
from copy import deepcopy
from collections import defaultdict
import threading
import queue
import sys

# Try to import terminal handling modules (Unix/Linux only)
try:
    import select
    import termios
    import tty
    UNIX_TERMINAL = True
except ImportError:
    UNIX_TERMINAL = False
    print("Warning: Advanced terminal features not available on this platform")

from tetris_engine import tEngine
from tetris_rewards import RewardCalculator

class TetrisTrainer:
    def __init__(self, q_table_path='q-ai/tetris_q_table.pkl'):
        self.q_table_path = q_table_path
        self.stats_path = 'q-ai/training_stats.json'
        self.checkpoint_dir = 'q-ai/checkpoints'
        
        # Create directories
        os.makedirs('q-ai', exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Q-Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Training components
        self.q_table = self.load_q_table()
        self.reward_calculator = RewardCalculator()
        self.actions = ['left', 'right', 'down', 'rotate', 'drop']
        
        # Training statistics
        self.training_stats = self.load_stats()
        self.episode_count = 0
        self.total_games = 0
        
        # Terminal state for proper restoration
        self.original_terminal_settings = None

    def safe_print(self, message, end='\n'):
        """Print with proper formatting for both normal and raw terminal modes"""
        if UNIX_TERMINAL and self.original_terminal_settings is not None:
            # In raw mode, use explicit carriage return + newline
            if end == '\n':
                print(f"\r{message}\r\n", end='', flush=True)
            else:
                print(f"\r{message}", end=end, flush=True)
        else:
            # Normal printing
            print(message, end=end, flush=True)

    def load_q_table(self):
        """Load existing Q-table or create new one"""
        try:
            with open(self.q_table_path, 'rb') as f:
                q_table = pickle.load(f)
                print(f"Loaded Q-table with {len(q_table)} entries")
                return defaultdict(float, q_table)
        except FileNotFoundError:
            print("Creating new Q-table")
            return defaultdict(float)
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            print("Creating new Q-table")
            return defaultdict(float)
    
    def save_q_table(self):
        """Save Q-table to disk"""
        try:
            q_table_dict = dict(self.q_table)
            with open(self.q_table_path, 'wb') as f:
                pickle.dump(q_table_dict, f)
            self.safe_print(f"Saved Q-table with {len(q_table_dict)} entries")
        except Exception as e:
            self.safe_print(f"Error saving Q-table: {e}")
    
    def load_stats(self):
        """Load training statistics"""
        try:
            with open(self.stats_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'episodes': [],
                'best_score': 0,
                'best_efficiency': 0,
                'total_training_time': 0
            }
        except Exception as e:
            print(f"Error loading stats: {e}")
            return {
                'episodes': [],
                'best_score': 0,
                'best_efficiency': 0,
                'total_training_time': 0
            }
    
    def save_stats(self):
        """Save training statistics"""
        try:
            with open(self.stats_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
        except Exception as e:
            self.safe_print(f"Error saving stats: {e}")
    
    def get_consistent_state(self, engine):
        """Get a consistent state representation from the engine"""
        try:
            board_state = tuple(tuple(row) for row in engine.board)
            piece_info = (
                str(engine.piece_type),
                int(engine.piece_x),
                int(engine.piece_y),
                tuple(tuple(row) for row in engine.piece)
            )
            state_features = self.reward_calculator.get_state_features(board_state, piece_info)
            return state_features, (board_state, piece_info)
            
        except Exception as e:
            print(f"Error getting state: {e}")
            default_board = tuple(tuple(0 for _ in range(10)) for _ in range(20))
            default_piece = ('I', 4, 0, ((1, 1, 1, 1),))
            default_features = ((0,) * 10, 0, 0, 'I', 4, 0)
            return default_features, (default_board, default_piece)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        try:
            if random.random() < self.epsilon:
                return random.choice(self.actions)
            else:
                q_values = []
                for action in self.actions:
                    key = (state, action)
                    q_values.append(self.q_table[key])
                
                max_q = max(q_values)
                best_actions = [action for action, q in zip(self.actions, q_values) if q == max_q]
                return random.choice(best_actions)
        except Exception as e:
            print(f"Error choosing action: {e}")
            return random.choice(self.actions)
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning algorithm"""
        try:
            current_key = (state, action)
            current_q = self.q_table[current_key]
            
            if done:
                target_q = reward
            else:
                future_q_values = []
                for a in self.actions:
                    future_key = (next_state, a)
                    future_q_values.append(self.q_table[future_key])
                
                max_future_q = max(future_q_values) if future_q_values else 0
                target_q = reward + self.discount_factor * max_future_q
            
            new_q = current_q + self.learning_rate * (target_q - current_q)
            self.q_table[current_key] = new_q
            
        except Exception as e:
            print(f"Error updating Q-table: {e}")
    
    def execute_action(self, engine, action):
        """Execute action on the engine safely"""
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
                while not engine.check_collision(dy=1):
                    engine.piece_y += 1
                engine.lock_piece()
                engine.clear_lines()
                engine.spawn_piece()
                engine.increase_level()
            return True
        except Exception as e:
            print(f"Error executing action {action}: {e}")
            return False
    
    def train_single_episode(self, max_moves=1000, verbose=False):
        """Train AI for one game episode"""
        try:
            engine = tEngine()
            total_reward = 0
            moves = 0
            
            current_state, current_full_state = self.get_consistent_state(engine)
            episode_start = time.time()
            
            while not engine.game_over and moves < max_moves:
                action = self.choose_action(current_state)
                prev_full_state = current_full_state
                prev_engine_state = deepcopy(engine)
                
                if not self.execute_action(engine, action):
                    break
                
                new_state, new_full_state = self.get_consistent_state(engine)
                reward = self.reward_calculator.calculate_reward(
                    prev_full_state, action, new_full_state, engine
                )
                total_reward += reward
                
                self.update_q_table(current_state, action, reward, new_state, engine.game_over)
                
                current_state = new_state
                current_full_state = new_full_state
                moves += 1
                
                if verbose and moves % 100 == 0:
                    self.safe_print(f"  Move {moves}, Score: {engine.score}, Reward: {total_reward:.1f}, Epsilon: {self.epsilon:.3f}")
            
            episode_time = time.time() - episode_start
            
            episode_stats = {
                'episode': self.episode_count,
                'score': engine.score,
                'moves': moves,
                'total_reward': total_reward,
                'efficiency': getattr(engine, 'efficiency', 0),
                'lines_cleared': engine.total_cleared,
                'level': engine.level,
                'time': episode_time,
                'epsilon': self.epsilon,
                'q_table_size': len(self.q_table)
            }
            
            self.training_stats['episodes'].append(episode_stats)
            
            if engine.score > self.training_stats['best_score']:
                self.training_stats['best_score'] = engine.score
            
            if hasattr(engine, 'efficiency') and engine.efficiency > self.training_stats['best_efficiency']:
                self.training_stats['best_efficiency'] = engine.efficiency
            
            self.training_stats['total_training_time'] += episode_time
            
            return episode_stats
            
        except Exception as e:
            print(f"Error in training episode: {e}")
            return {
                'episode': self.episode_count,
                'score': 0,
                'moves': 0,
                'total_reward': 0,
                'efficiency': 0,
                'lines_cleared': 0,
                'level': 1,
                'time': 0,
                'epsilon': self.epsilon,
                'q_table_size': len(self.q_table)
            }
    
    def train_batch(self, num_episodes, save_interval=100, verbose=True):
        """Train AI for multiple episodes with proper formatting"""
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Initial epsilon: {self.epsilon}")
        
        batch_start = time.time()
        
        for episode in range(num_episodes):
            self.episode_count += 1
            
            try:
                episode_stats = self.train_single_episode(verbose=verbose and episode % 10 == 0)
                
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                if verbose and (episode + 1) % 10 == 0:
                    recent_episodes = self.training_stats['episodes'][-10:]
                    if recent_episodes:
                        recent_scores = [ep['score'] for ep in recent_episodes]
                        avg_score = sum(recent_scores) / len(recent_scores)
                        print(f"Episode {episode + 1}/{num_episodes}: "
                              f"Score={episode_stats['score']}, "
                              f"Avg(10)={avg_score:.1f}, "
                              f"Efficiency={episode_stats['efficiency']:.1f}%, "
                              f"Q-size={len(self.q_table)}")
                
                if (episode + 1) % save_interval == 0:
                    self.save_checkpoint(episode + 1)
                    
            except Exception as e:
                print(f"Error in episode {episode + 1}: {e}")
                continue
        
        batch_time = time.time() - batch_start
        print(f"\nTraining completed in {batch_time:.1f} seconds")
        print(f"Final epsilon: {self.epsilon:.4f}")
        print(f"Q-table size: {len(self.q_table)}")
        
        self.save_q_table()
        self.save_stats()
    
    def train_interactive(self, save_interval=100, verbose=True):
        """Interactive training with improved terminal handling"""
        self.safe_print("Starting interactive training...")
        self.safe_print("Commands during training:")
        self.safe_print("  's' - Save current progress")
        self.safe_print("  'q' - Quit training")
        self.safe_print("  'e' - Evaluate current performance")
        self.safe_print("  'p' - Pause/Resume training")
        self.safe_print("  '+' - Increase save frequency")
        self.safe_print("  '-' - Decrease save frequency")
        
        if not UNIX_TERMINAL:
            self.safe_print("Note: Interactive commands not available on this platform")
            self.safe_print("Training will run continuously. Press Ctrl+C to stop.")
            input("Press Enter to start...")
            # Fall back to simple continuous training
            try:
                while True:
                    self.episode_count += 1
                    episode_stats = self.train_single_episode(verbose=False)
                    
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay
                    
                    if verbose and self.episode_count % 10 == 0:
                        recent_episodes = self.training_stats['episodes'][-10:]
                        if recent_episodes:
                            recent_scores = [ep['score'] for ep in recent_episodes]
                            avg_score = sum(recent_scores) / len(recent_scores)
                            elapsed = time.time()
                            print(f"Ep {self.episode_count}: Score={episode_stats['score']}, "
                                  f"Avg={avg_score:.1f}, Eff={episode_stats['efficiency']:.1f}%, "
                                  f"Q={len(self.q_table)}")
                    
                    if self.episode_count % save_interval == 0:
                        self.save_checkpoint(self.episode_count)
                        self.save_q_table()
                        self.save_stats()
                        
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
                        self.save_checkpoint(self.episode_count)
                        self.save_q_table()
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

                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay

                    # Show progress on same line
                    if verbose and self.episode_count % 5 == 0:  # More frequent updates
                        recent_episodes = self.training_stats['episodes'][-10:]
                        if recent_episodes:
                            recent_scores = [ep['score'] for ep in recent_episodes]
                            avg_score = sum(recent_scores) / len(recent_scores)
                            elapsed = time.time() - batch_start
                            
                            progress_line = (f"Ep {self.episode_count}: "
                                           f"Score={episode_stats['score']}, "
                                           f"Avg={avg_score:.1f}, "
                                           f"Eff={episode_stats['efficiency']:.1f}%, "
                                           f"Q={len(self.q_table)}, "
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
                        self.save_checkpoint(self.episode_count)
                        self.save_q_table()
                        self.save_stats()
                        episodes_since_save = 0
                        last_progress_line = ""  # Reset progress line after save message
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
    
    def save_checkpoint(self, episode):
        """Save training checkpoint"""
        try:
            checkpoint_path = f"{self.checkpoint_dir}/checkpoint_ep{episode}.pkl"
            checkpoint_data = {
                'q_table': dict(self.q_table),
                'episode': episode,
                'epsilon': self.epsilon,
                'training_stats': self.training_stats
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
        except Exception as e:
            self.safe_print(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.q_table = defaultdict(float, checkpoint['q_table'])
            self.episode_count = checkpoint['episode']
            self.epsilon = checkpoint['epsilon']
            self.training_stats = checkpoint['training_stats']
            
            print(f"Loaded checkpoint from episode {self.episode_count}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    def evaluate_performance(self, num_games=10):
        """Evaluate current AI performance without learning"""
        # Clear any progress lines first
        self.safe_print(f"Evaluating AI performance over {num_games} games...")
        
        old_epsilon = self.epsilon
        self.epsilon = 0
        
        scores = []
        efficiencies = []
        
        for game in range(num_games):
            stats = self.train_single_episode(max_moves=2000, verbose=False)
            scores.append(stats['score'])
            efficiencies.append(stats['efficiency'])
            
            if (game + 1) % 5 == 0:
                self.safe_print(f"  Game {game + 1}/{num_games}: Score={stats['score']}, Efficiency={stats['efficiency']:.1f}%")
        
        self.epsilon = old_epsilon
        
        avg_score = sum(scores) / len(scores) if scores else 0
        avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0
        
        self.safe_print("Evaluation Results:")
        self.safe_print(f"Average Score: {avg_score:.1f}")
        self.safe_print(f"Best Score: {max(scores) if scores else 0}")
        self.safe_print(f"Average Efficiency: {avg_efficiency:.1f}%")
        self.safe_print(f"Best Efficiency: {max(efficiencies) if efficiencies else 0:.1f}%")
        
        return {
            'avg_score': avg_score,
            'best_score': max(scores) if scores else 0,
            'avg_efficiency': avg_efficiency,
            'best_efficiency': max(efficiencies) if efficiencies else 0,
            'scores': scores,
            'efficiencies': efficiencies
        }

def main():
    """Main training interface"""
    trainer = TetrisTrainer()
    
    print("Tetris AI Trainer")
    print("=================")
    print("Commands:")
    print("  train <episodes> - Train for specified number of episodes")
    print("  eval <games>     - Evaluate current AI performance")
    print("  quick            - Quick training session (100 episodes)")
    print("  long             - Long training session (1000 episodes)")
    print("  continuous       - Train continuously with interactive controls")
    print("  auto <save_freq> - Auto-train with custom save frequency")
    print("  stats            - Show training statistics")
    print("  quit             - Exit trainer")
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
            
            elif command[0] == 'quick':
                trainer.train_batch(100)
            
            elif command[0] == 'long':
                trainer.train_batch(1000)
            
            elif command[0] == 'continuous':
                trainer.train_interactive()
            
            elif command[0] == 'auto':
                save_freq = int(command[1]) if len(command) > 1 else 100
                trainer.train_interactive(save_interval=save_freq)
            
            elif command[0] == 'stats':
                stats = trainer.training_stats
                print(f"Total episodes: {len(stats['episodes'])}")
                print(f"Best score: {stats['best_score']}")
                print(f"Best efficiency: {stats['best_efficiency']:.1f}%")
                print(f"Q-table size: {len(trainer.q_table)}")
                print(f"Current epsilon: {trainer.epsilon:.4f}")
            
            else:
                print("Unknown command")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted")
            break
        except ValueError:
            print("Invalid number format")
        except Exception as e:
            print(f"Error: {e}")
    
    print("Trainer finished!")

if __name__ == "__main__":
    main()