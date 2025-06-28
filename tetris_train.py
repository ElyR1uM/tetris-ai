"""
Optimized DQN Trainer for Tetris AI with Reward-Based Epsilon Decay (RBED)
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
        
        # Traditional epsilon decay parameters (kept for compatibility)
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        
        # Reward-Based Epsilon Decay (RBED) parameters
        self.use_rbed = True
        self.rbed_epsilon = 1.0
        self.rbed_epsilon_min = 0.05
        self.rbed_reward_threshold = 0  # Start with 0 for Tetris (can go negative)
        self.rbed_reward_increment = 50  # Increase threshold by 50 points each time
        self.rbed_epsilon_delta = 0.05  # Decrease epsilon by this amount when threshold is met
        self.rbed_smoothing_window = 10  # Use average of last N episodes for stability
        
        # RBED tracking
        self.rbed_recent_rewards = deque(maxlen=self.rbed_smoothing_window)
        self.rbed_thresholds_met = 0
        self.rbed_last_decay_episode = 0
        
        # Other training parameters
        self.batch_size = 64
        self.memory_size = memory_size
        self.target_update_freq = 1000
        self.train_freq = 4
        
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
        
        # Continuous training control
        self.continuous_training = False
        self.stop_training = False
        
    def configure_rbed(self, initial_threshold=0, reward_increment=50, epsilon_delta=0.05, 
                      smoothing_window=10, min_epsilon=0.05):
        """Configure Reward-Based Epsilon Decay parameters"""
        self.rbed_reward_threshold = initial_threshold
        self.rbed_reward_increment = reward_increment
        self.rbed_epsilon_delta = epsilon_delta
        self.rbed_smoothing_window = smoothing_window
        self.rbed_epsilon_min = min_epsilon
        
        # Reinitialize tracking
        self.rbed_recent_rewards = deque(maxlen=self.rbed_smoothing_window)
        self.rbed_thresholds_met = 0
        self.rbed_last_decay_episode = 0
        
        print(f"RBED configured:")
        print(f"  Initial threshold: {initial_threshold}")
        print(f"  Reward increment: {reward_increment}")
        print(f"  Epsilon delta: {epsilon_delta}")
        print(f"  Smoothing window: {smoothing_window}")
        print(f"  Minimum epsilon: {min_epsilon}")
    
    def get_current_epsilon(self):
        """Get the current epsilon value based on selected strategy"""
        if self.use_rbed:
            return max(self.rbed_epsilon, self.rbed_epsilon_min)
        else:
            return max(self.epsilon, self.epsilon_min)
    
    def update_epsilon_rbed(self, episode_reward):
        """Update epsilon using Reward-Based Epsilon Decay"""
        # Add reward to recent rewards buffer
        self.rbed_recent_rewards.append(episode_reward)
        
        # Only proceed if we have enough samples for smoothing
        if len(self.rbed_recent_rewards) < self.rbed_smoothing_window:
            return False
        
        # Calculate smoothed reward (average of recent episodes)
        smoothed_reward = np.mean(self.rbed_recent_rewards)
        
        # Check if we should decay epsilon
        if (self.rbed_epsilon > self.rbed_epsilon_min and 
            smoothed_reward >= self.rbed_reward_threshold):
            
            # Decay epsilon
            old_epsilon = self.rbed_epsilon
            self.rbed_epsilon = max(
                self.rbed_epsilon - self.rbed_epsilon_delta,
                self.rbed_epsilon_min
            )
            
            # Update threshold for next decay
            self.rbed_reward_threshold += self.rbed_reward_increment
            self.rbed_thresholds_met += 1
            self.rbed_last_decay_episode = self.episode_count
            
            print(f"\n🎯 RBED Epsilon Decay Triggered!")
            print(f"  Episode: {self.episode_count}")
            print(f"  Smoothed reward: {smoothed_reward:.1f}")
            print(f"  Threshold met: {self.rbed_reward_threshold - self.rbed_reward_increment}")
            print(f"  Epsilon: {old_epsilon:.4f} → {self.rbed_epsilon:.4f}")
            print(f"  Next threshold: {self.rbed_reward_threshold}")
            print(f"  Thresholds met so far: {self.rbed_thresholds_met}")
            
            return True
        
        return False
    
    def update_epsilon_traditional(self):
        """Update epsilon using traditional exponential decay"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def build_optimized_model(self):
        """Build optimized CNN model with fixed architecture to prevent dimension errors"""
        model = models.Sequential([
            layers.Input(shape=(self.board_height, self.board_width, 1)),
            
            # First conv layer with padding to maintain reasonable dimensions
            layers.Conv2D(32, (4, 4), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),  # 20x10 -> 10x5
            
            # Second conv layer with smaller kernel
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),  # 10x5 -> 5x2 (rounded down)
            
            # Third conv layer optimized for remaining dimensions
            layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
            
            # Global average pooling to handle variable dimensions gracefully
            layers.GlobalAveragePooling2D(),
            
            # Dense layers for decision making
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
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
                print(f"Model parameters: {self.q_network.count_params()}") # type: ignore
            else:
                print("No existing model found, starting with fresh model")
                print(f"Model parameters: {self.q_network.count_params()}") # type: ignore
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with fresh model")
    
    def save_model(self):
        """Save the trained model"""
        try:
            self.q_network.save(self.model_path) # type: ignore
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights()) # type: ignore
    
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
        """Choose action using epsilon-greedy policy with current epsilon"""
        current_epsilon = self.get_current_epsilon()
        
        if np.random.random() <= current_epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Batch prediction for efficiency
        state_batch = np.expand_dims(state, axis=0)
        q_values = self.q_network.predict(state_batch, verbose=0)[0] # type: ignore
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
        current_q_values = self.q_network.predict(states, verbose=0) # type: ignore
        next_q_values = self.target_network.predict(next_states, verbose=0) # type: ignore
        
        # Vectorized Q-value updates
        target_q_values = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Single training step
        self.q_network.fit(states, target_q_values, epochs=1, verbose=0) # type: ignore
    
    def train_single_episode(self, max_moves=2000, verbose=False):
        """Optimized single episode training with RBED"""
        engine = tEngine()
        total_reward = 0
        moves = 0
        
        current_state = self.preprocess_state(engine)
        episode_start = time.time()
        
        while not engine.game_over and moves < max_moves:
            # Check for stop signal in continuous mode
            if self.continuous_training and self.stop_training:
                break
                
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
        
        # Update epsilon based on selected strategy
        if self.use_rbed:
            rbed_triggered = self.update_epsilon_rbed(engine.score)
        else:
            self.update_epsilon_traditional()
            rbed_triggered = False
        
        episode_time = time.time() - episode_start
        current_epsilon = self.get_current_epsilon()
        
        episode_stats = {
            'episode': self.episode_count,
            'score': engine.score,
            'moves': moves,
            'total_reward': total_reward,
            'lines_cleared': engine.total_cleared,
            'level': engine.level,
            'time': episode_time,
            'epsilon': current_epsilon,
            'memory_size': len(self.memory),
            'rbed_threshold': self.rbed_reward_threshold if self.use_rbed else None,
            'rbed_triggered': rbed_triggered,
            'rbed_thresholds_met': self.rbed_thresholds_met if self.use_rbed else None
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
        """Optimized batch training with RBED performance tracking"""
        decay_strategy = "RBED" if self.use_rbed else "Traditional"
        print(f"Starting optimized DQN training for {num_episodes} episodes...")
        print(f"Epsilon decay strategy: {decay_strategy}")
        print(f"Model parameters: {self.q_network.count_params()}") # type: ignore
        
        if self.use_rbed:
            print(f"RBED Settings:")
            print(f"  Current threshold: {self.rbed_reward_threshold}")
            print(f"  Reward increment: {self.rbed_reward_increment}")
            print(f"  Epsilon delta: {self.rbed_epsilon_delta}")
            print(f"  Smoothing window: {self.rbed_smoothing_window}")
            print(f"  Current epsilon: {self.rbed_epsilon:.4f}")
        
        batch_start = time.time()
        
        for episode in range(num_episodes):
            # Check for stop signal in continuous mode
            if self.continuous_training and self.stop_training:
                print("Training stopped by user")
                break
                
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
                    rbed_info = ""
                    if self.use_rbed:
                        smoothed = np.mean(self.rbed_recent_rewards) if self.rbed_recent_rewards else 0
                        rbed_info = f", Thr={self.rbed_reward_threshold}, Sm={smoothed:.1f}"
                    
                    print(f"Episode {self.episode_count}: "
                          f"Score={episode_stats['score']}, "
                          f"Avg100={avg_score:.1f}, "
                          f"Lines={episode_stats['lines_cleared']}, "
                          f"ε={episode_stats['epsilon']:.4f}"
                          f"{rbed_info}, "
                          f"Time={episode_stats['time']:.2f}s")
                
                if (episode + 1) % save_interval == 0:
                    self.save_model()
                    self.save_stats()
                    print(f"Checkpoint saved at episode {self.episode_count}")
                    
            except Exception as e:
                print(f"Error in episode {self.episode_count}: {e}")
                continue
        
        batch_time = time.time() - batch_start
        avg_time_per_episode = batch_time / num_episodes if num_episodes > 0 else 0
        
        print(f"\nTraining completed in {batch_time:.1f} seconds")
        print(f"Average time per episode: {avg_time_per_episode:.2f} seconds")
        print(f"Average score (last 100): {np.mean(self.last_100_scores):.1f}")
        print(f"Best score: {max(self.episode_scores) if self.episode_scores else 0}")
        
        if self.use_rbed:
            print(f"RBED thresholds met: {self.rbed_thresholds_met}")
            print(f"Final epsilon: {self.rbed_epsilon:.4f}")
            print(f"Current threshold: {self.rbed_reward_threshold}")
        
        self.save_model()
        self.save_stats()
        self.save_high_scores()
    
    def train_continuous(self, save_interval=50, status_interval=10):
        """Continuous training mode with RBED - train until stopped"""
        decay_strategy = "RBED" if self.use_rbed else "Traditional"
        print("Starting continuous training mode...")
        print(f"Epsilon decay strategy: {decay_strategy}")
        print("Press Ctrl+C to stop training gracefully")
        print(f"Model parameters: {self.q_network.count_params()}") # type: ignore
        print("Status updates every", status_interval, "episodes")
        print("Auto-save every", save_interval, "episodes")
        
        if self.use_rbed:
            print(f"RBED Settings:")
            print(f"  Current threshold: {self.rbed_reward_threshold}")
            print(f"  Current epsilon: {self.rbed_epsilon:.4f}")
        print()
        
        self.continuous_training = True
        self.stop_training = False
        
        start_time = time.time()
        episode_in_session = 0
        
        try:
            while not self.stop_training:
                episode_in_session += 1
                self.episode_count += 1
                
                try:
                    episode_stats = self.train_single_episode()
                    
                    # Update high scores every 100 episodes
                    self.update_high_scores(episode_stats['score'], self.episode_count)
                    
                    # Save training stats
                    if not hasattr(self, 'training_stats'):
                        self.training_stats = {'episodes': []}
                    self.training_stats['episodes'].append(episode_stats)
                    
                    # Status updates
                    if episode_in_session % status_interval == 0:
                        avg_score = np.mean(self.last_100_scores) if self.last_100_scores else 0
                        session_time = time.time() - start_time
                        avg_time = session_time / episode_in_session
                        
                        rbed_info = ""
                        if self.use_rbed:
                            smoothed = np.mean(self.rbed_recent_rewards) if self.rbed_recent_rewards else 0
                            rbed_info = f", Thr={self.rbed_reward_threshold}, Sm={smoothed:.1f}"
                        
                        print(f"Episode {self.episode_count} (Session: {episode_in_session}): "
                              f"Score={episode_stats['score']}, "
                              f"Avg100={avg_score:.1f}, "
                              f"Lines={episode_stats['lines_cleared']}, "
                              f"ε={episode_stats['epsilon']:.4f}"
                              f"{rbed_info}, "
                              f"Time={avg_time:.2f}s/ep")
                    
                    # Auto-save
                    if episode_in_session % save_interval == 0:
                        self.save_model()
                        self.save_stats()
                        self.save_high_scores()
                        session_time = time.time() - start_time
                        print(f"\n--- Checkpoint saved at episode {self.episode_count} ---")
                        print(f"Session time: {session_time/3600:.2f} hours")
                        print(f"Episodes this session: {episode_in_session}")
                        print(f"Average time per episode: {session_time/episode_in_session:.2f}s")
                        if self.last_100_scores:
                            print(f"Current average score: {np.mean(self.last_100_scores):.1f}")
                        if self.use_rbed:
                            print(f"RBED thresholds met: {self.rbed_thresholds_met}")
                            print(f"Current epsilon: {self.rbed_epsilon:.4f}")
                        print("---")
                        
                except Exception as e:
                    print(f"Error in episode {self.episode_count}: {e}")
                    continue
                    
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            
        finally:
            self.stop_training = True
            self.continuous_training = False
            
            # Final save
            self.save_model()
            self.save_stats()
            self.save_high_scores()
            
            session_time = time.time() - start_time
            print(f"\nContinuous training session completed:")
            print(f"Total episodes this session: {episode_in_session}")
            print(f"Total training time: {session_time/3600:.2f} hours")
            if episode_in_session > 0:
                print(f"Average time per episode: {session_time/episode_in_session:.2f} seconds")
            if self.last_100_scores:
                print(f"Final average score (last 100): {np.mean(self.last_100_scores):.1f}")
            print(f"Best score ever: {max(self.episode_scores) if self.episode_scores else 0}")
            if self.use_rbed:
                print(f"Final RBED thresholds met: {self.rbed_thresholds_met}")
                print(f"Final epsilon: {self.rbed_epsilon:.4f}")
    
    def switch_epsilon_strategy(self, use_rbed=True):
        """Switch between RBED and traditional epsilon decay"""
        self.use_rbed = use_rbed
        strategy = "RBED" if use_rbed else "Traditional"
        print(f"Switched to {strategy} epsilon decay strategy")
        
        if use_rbed:
            print(f"Current RBED settings:")
            print(f"  Epsilon: {self.rbed_epsilon:.4f}")
            print(f"  Threshold: {self.rbed_reward_threshold}")
            print(f"  Thresholds met: {self.rbed_thresholds_met}")
        else:
            print(f"Current traditional settings:")
            print(f"  Epsilon: {self.epsilon:.4f}")
            print(f"  Decay rate: {self.epsilon_decay}")
    
    def reset_rbed(self, reset_epsilon=True):
        """Reset RBED parameters"""
        if reset_epsilon:
            self.rbed_epsilon = 1.0
        self.rbed_reward_threshold = 0
        self.rbed_recent_rewards.clear()
        self.rbed_thresholds_met = 0
        self.rbed_last_decay_episode = 0
        
        print("RBED parameters reset")
        print(f"  Epsilon: {self.rbed_epsilon:.4f}")
        print(f"  Threshold: {self.rbed_reward_threshold}")
        print(f"  Thresholds met: {self.rbed_thresholds_met}")
    
    def get_rbed_status(self):
        """Display detailed RBED status"""
        if not self.use_rbed:
            print("RBED is not currently active.")
            return
        
        smoothed = np.mean(self.rbed_recent_rewards) if self.rbed_recent_rewards else 0
        print("\nRBED Status:")
        print(f"  Current epsilon: {self.rbed_epsilon:.4f}")
        print(f"  Minimum epsilon: {self.rbed_epsilon_min}")
        print(f"  Current threshold: {self.rbed_reward_threshold}")
        print(f"  Reward increment: {self.rbed_reward_increment}")
        print(f"  Epsilon delta: {self.rbed_epsilon_delta}")
        print(f"  Smoothing window: {self.rbed_smoothing_window}")
        print(f"  Smoothed reward: {smoothed:.1f}")
        print(f"  Thresholds met: {self.rbed_thresholds_met}")
        print(f"  Last decay at episode: {self.rbed_last_decay_episode}")

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
    """Command-line interface for RBED-enabled Tetris DQN training"""
    trainer = AgentTrainer()

    print("RBED-Enabled DQN Tetris Trainer")
    print("===============================")
    print("Commands:")
    print("  train <episodes>      - Train for specified number of episodes")
    print("  continuous            - Start continuous training")
    print("  stop                  - Stop continuous training")
    print("  switch_rbed <on/off>  - Enable or disable RBED")
    print("  reset_rbed            - Reset RBED parameters")
    print("  rbed_status           - Show RBED status")
    print("  scores                - Show high scores")
    print("  epsilon               - Show current epsilon")
    print("  quick                 - Quick 100 episode train")
    print("  quit                  - Exit trainer")
    print()

    while True:
        try:
            command = input("Command> ").strip().lower().split()

            if not command:
                continue

            if command[0] == "quit":
                break
            elif command[0] == "train":
                episodes = int(command[1]) if len(command) > 1 else 100
                trainer.train_batch(episodes)
            elif command[0] == "continuous":
                trainer.train_continuous()
            elif command[0] == "stop":
                trainer.stop_training = True
                print("Stop signal sent.")
            elif command[0] == "switch_rbed":
                if len(command) > 1 and command[1] in ['on', 'off']:
                    trainer.switch_epsilon_strategy(use_rbed=(command[1] == 'on'))
                else:
                    print("Usage: switch_rbed <on/off>")
            elif command[0] == "reset_rbed":
                trainer.reset_rbed()
            elif command[0] == "rbed_status":
                trainer.get_rbed_status()
            elif command[0] == "epsilon":
                print(f"Current epsilon: {trainer.get_current_epsilon():.4f}")
            elif command[0] == "scores":
                if trainer.high_scores:
                    print("High Scores:")
                    for entry in trainer.high_scores[-10:]:
                        print(f"  Episode {entry['episode']}: {entry['score']}")
                else:
                    print("No high scores yet.")
            elif command[0] == "quick":
                trainer.train_batch(100)
            else:
                print("Unknown command.")
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
