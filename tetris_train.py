"""Trainer for qAgent"""

# Keras is the main library used for the model training
import tensorflow as tf
import keras
from keras import models, layers
from keras.optimizers import Adam

import pickle
import random
import json
import time
import os
import threading
import queue
import sys
from copy import deepcopy
from collections import defaultdict

from tetris_engine import tEngine
from tetris_rewards import RewardCalculator

def build_model(input_shape, num_actions):
    """Keras-based Model template"""
    model = models.Sequential([
        layers.Input(shape= input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    # Type checker will whine if next line is not ignored
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse') # type: ignore
    return model

class AgentTrainer:
    """Trainer for qAgent"""
    def __init__(self, q_table_path='q-ai/q_table.pkl'):
        self.q_table_path = q_table_path
        self.checkpoint_dir = 'q-ai/checkpoints'

        # Create directories
        os.makedirs('q-ai', exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training Components
        self.q_table = self.load_q_table()
        self.reward_calculator = RewardCalculator()
        self.actions = ['left', 'right', 'drop', 'rotate']

    def load_q_table(self):
        try:
            with open(self.q_table_path, 'rb') as f:
                q_table = pickle.load(f)
                print(f"Loaded Q-Table with {len(q_table)} entries")
                return defaultdict(float, q_table)
        except FileNotFoundError:
            print("Creating new Q-Table")
            return defaultdict(float)
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            return defaultdict(float)
    
    def save_q_table(self):
        """Save Q-Table to disk"""
        try:
            q_table_dict = dict(self.q_table)
            with open(self.q_table_path, 'wb') as f:
                pickle.dump(q_table_dict, f)
            print(f"Saved Q-table with {len(q_table_dict)} entries")
        except Exception as e:
            print(f"Error saving Q-table: {e}")
