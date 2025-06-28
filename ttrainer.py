"""
Optimized DQN Trainer for Tetris AI with Performance Improvements and High Score Tracking
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

class AgentTrainer:
    def __init__(self, model_path='q-ai/dqn_model.h5', memory_size=50000):
        self.model_path = model_path
        self.stats_path = 'q-ai/dqn_training_stats.json'
        self.checkpoint_dir = 'q-ai/dqn_checkpoints'
        self.scores_path = 'out/q-scores.json'
        
        # Create directories
        os.makedirs('q-ai', exist_ok=True)
        os.makedirs('out', exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Optimized DQN hyperparameters
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64  # Increased for better efficiency
        self.memory_size = memory_size
        self.target_update_freq = 1000
        self.train_freq = 4  # Train every 4 steps instead of every step
        
        # Game parameters
        self.board_height = 20
        self.board_width = 10
        self.actions = ['left', 'right', 'down', 'rotate', 'drop']
        self.num_actions = len(self.actions)
        
        # Cell type mapping
        self.cell_mapping = {
            0: 0.0, '0': 0.0, ' ': 0.0,
            'I': 1.0, 'O': 2.0, 'T': 3.0, 'S': 4.0,
            'Z': 5.0, 'J': 6.0, 'L': 7.0,
        }
        
        # Training components
        self.memory = deque(maxlen=memory_size)
        self.reward_calculator = RewardCalculator()
        
        # Neural networks
        self.q_network = self.build_optimized_model()
        self.target_network = self.build_optimized_model()
        self.update_target_network()
        
        # Load existing model if available
        self.load_model()
        
        # Training statistics
        self.training_stats = self.load_stats()
        self.high_scores = self.load_high_scores()
        self.episode_count = 0
        self.step_count = 0
        
        # Performance tracking
        self.episode_scores = []
        self.last_100_scores = deque(maxlen=100)
        
    def build_optimized_model(self):
        """Build optimized CNN model with fewer parameters"""
        model = models.Sequential([
            layers.Input(shape=(self.board_height, self.board_width, 1)),
            
            # Smaller, more efficient conv layers
            layers.Conv2D(16, (4, 4), activation='relu', strides=(2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.Conv2D(32, (3, 3), activation='relu'),
            
            # Flatten instead of global pooling for speed
            layers.Flatten(),
            
            # Smaller dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            
            # Output layer
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
                print(f"Model parameters: {self.q_network.count_params()}")
            else:
                print("No existing model found, starting with fresh model")
                print(f"Model parameters: {self.q_network.count_params()}")
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
        """Optimized state preprocessing with caching"""
        try:
            # Create board representation
            board = np.zeros((self.board_height, self.board_width), dtype=np.float32)
            
            # Vectorized board conversion
            for y in range(self.board_height):
                for x in range(self.board_width):
                    cell_value = engine.board[y][x]
                    board[y, x] = self.cell_mapping.get(cell_value, 0.0)
            
            # Add current piece
            if hasattr(engine, 'piece') and engine.piece is not None:
                piece_y, piece_x = engine.piece_y, engine.piece_x
                piece = np.array(engine.piece)
                piece_type = getattr(engine, 'piece_type', 'T')
                piece_value = self.cell_mapping.get(piece_type, 8.0)
                
                for py in range(piece.shape[0]):
                    for px in range(piece.shape[1]):
                        if piece[py, px] != 0:
                            board_y, board_x = piece_y + py, piece_x + px
                            if (0 <= board_y < self.board_height and 
                                0 <= board_x < self.board_width):
                                board[board_y, board_x] = piece_value
            
            # Normalize and reshape
            board = board / 8.0
            state = board.reshape(self.board_height, self.board_width, 1)
            
            return state
            
        except Exception as e:
            print(f"Error preprocessing state: {e}")
            return np.zeros((self.board_height, self.board_width, 1), dtype=np.float32)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Batch prediction for efficiency
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
            return False
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Optimized training with larger batches"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Batch predictions
        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Vectorized Q-value updates
        target_q_values = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Single training step
        self.q_network.fit(states, target_q_values, epochs=1, verbose=0)
    
    def train_single_episode(self, max_moves=2000, verbose=False):
        """Optimized single episode training"""
        engine = tEngine()
        total_reward = 0
        moves = 0
        
        current_state = self.preprocess_state(engine)
        episode_start = time.time()
        
        while not engine.game_over and moves < max_moves:
            # Choose and execute action
            action_idx = self.choose_action(current_state)
            
            # Store previous state
            prev_engine_state = deepcopy(engine)
            prev_full_state = self.get_full_state(engine)
            
            # Execute action
            if not self.execute_action(engine, action_idx):
                break
            
            # Get new state and reward
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
            
            # Train less frequently for speed
            if len(self.memory) > self.batch_size and self.step_count % self.train_freq == 0:
                self.replay()
            
            # Update target network
            if self.step_count % self.target_update_freq == 0:
                self.update_target_network()
            
            current_state = next_state
            moves += 1
            self.step_count += 1
        
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
        
        # Track scores for performance analysis
        self.episode_scores.append(engine.score)
        self.last_100_scores.append(engine.score)
        
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
    
    def load_high_scores(self):
        """Load high scores from file"""
        try:
            with open(self.scores_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error loading high scores: {e}")
            return []
    
    def save_high_scores(self):
        """Save high scores to file"""
        try:
            with open(self.scores_path, 'w') as f:
                json.dump(self.high_scores, f, indent=2)
        except Exception as e:
            print(f"Error saving high scores: {e}")
    
    def update_high_scores(self, score, episode):
        """Update high scores every 100 episodes"""
        if episode % 100 == 0:
            # Get the best score from the last 100 episodes
            recent_scores = self.episode_scores[-100:] if len(self.episode_scores) >= 100 else self.episode_scores
            best_recent_score = max(recent_scores) if recent_scores else 0
            
            high_score_entry = {
                "score": best_recent_score,
                "episode": episode
            }
            
            self.high_scores.append(high_score_entry)
            self.save_high_scores()
            
            print(f"High score updated: {best_recent_score} at episode {episode}")
    
    def train_batch(self, num_episodes, save_interval=50, verbose=True):
        """Optimized batch training with performance tracking"""
        print(f"Starting optimized DQN training for {num_episodes} episodes...")
        print(f"Model parameters: {self.q_network.count_params()}")
        
        batch_start = time.time()
        
        for episode in range(num_episodes):
            self.episode_count += 1
            
            try:
                episode_stats = self.train_single_episode(verbose=verbose and episode % 50 == 0)
                
                # Update high scores every 100 episodes
                self.update_high_scores(episode_stats['score'], self.episode_count)
                
                # Save training stats
                if not hasattr(self, 'training_stats'):
                    self.training_stats = {'episodes': []}
                self.training_stats['episodes'].append(episode_stats)
                
                if verbose and (episode + 1) % 10 == 0:
                    avg_score = np.mean(self.last_100_scores) if self.last_100_scores else 0
                    print(f"Episode {self.episode_count}: "
                          f"Score={episode_stats['score']}, "
                          f"Avg100={avg_score:.1f}, "
                          f"Lines={episode_stats['lines_cleared']}, "
                          f"ε={self.epsilon:.3f}, "
                          f"Time={episode_stats['time']:.2f}s")
                
                if (episode + 1) % save_interval == 0:
                    self.save_model()
                    self.save_stats()
                    print(f"Checkpoint saved at episode {self.episode_count}")
                    
            except Exception as e:
                print(f"Error in episode {self.episode_count}: {e}")
                continue
        
        batch_time = time.time() - batch_start
        avg_time_per_episode = batch_time / num_episodes
        
        print(f"\nTraining completed in {batch_time:.1f} seconds")
        print(f"Average time per episode: {avg_time_per_episode:.2f} seconds")
        print(f"Average score (last 100): {np.mean(self.last_100_scores):.1f}")
        print(f"Best score: {max(self.episode_scores) if self.episode_scores else 0}")
        
        self.save_model()
        self.save_stats()
        self.save_high_scores()
    
    def analyze_performance(self):
        """Analyze when AI might start scoring consistently"""
        if len(self.episode_scores) < 100:
            print("Need at least 100 episodes for meaningful analysis")
            return
        
        # Calculate rolling averages
        window_size = 100
        rolling_averages = []
        
        for i in range(window_size, len(self.episode_scores) + 1):
            window = self.episode_scores[i-window_size:i]
            rolling_averages.append(np.mean(window))
        
        # Find when scores consistently stay above 0
        positive_streaks = []
        current_streak = 0
        
        for avg in rolling_averages:
            if avg > 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    positive_streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            positive_streaks.append(current_streak)
        
        print("\nPerformance Analysis:")
        print(f"Total episodes: {len(self.episode_scores)}")
        print(f"Current average (last 100): {np.mean(self.last_100_scores):.2f}")
        print(f"Best score achieved: {max(self.episode_scores)}")
        print(f"Episodes with score > 0: {sum(1 for s in self.episode_scores if s > 0)}")
        
        if positive_streaks:
            print(f"Longest streak of positive averages: {max(positive_streaks)} episodes")
        
        # Predict when consistent positive scores might occur
        if len(rolling_averages) > 10:
            recent_trend = np.polyfit(range(len(rolling_averages)), rolling_averages, 1)[0]
            if recent_trend > 0:
                episodes_to_positive = max(0, int(-rolling_averages[-1] / recent_trend))
                print(f"Estimated episodes until consistent positive scores: ~{episodes_to_positive}")
            else:
                print("No positive trend detected yet")
    
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
    """Main training interface with performance analysis"""
    trainer = AgentTrainer()
    
    print("Optimized DQN Tetris AI Trainer")
    print("===============================")
    print("Commands:")
    print("  train <episodes>  - Train for specified episodes")
    print("  eval <games>      - Evaluate performance")
    print("  analyze           - Analyze performance trends")
    print("  scores            - Show high scores")
    print("  quick             - Quick training (100 episodes)")
    print("  benchmark         - Speed benchmark (10 episodes)")
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
            
            elif command[0] == 'analyze':
                trainer.analyze_performance()
            
            elif command[0] == 'scores':
                if trainer.high_scores:
                    print("High Scores (every 100 episodes):")
                    for entry in trainer.high_scores[-10:]:  # Show last 10
                        print(f"Episode {entry['episode']}: {entry['score']}")
                else:
                    print("No high scores recorded yet")
            
            elif command[0] == 'benchmark':
                print("Running speed benchmark...")
                start_time = time.time()
                trainer.train_batch(10, verbose=False)
                end_time = time.time()
                print(f"10 episodes completed in {end_time - start_time:.2f} seconds")
                print(f"Average: {(end_time - start_time) / 10:.2f} seconds per episode")
            
            elif command[0] == 'quick':
                trainer.train_batch(100)
            
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