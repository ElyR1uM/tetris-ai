import keras
from keras import initializers
from keras.models import Sequential, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import sys

class Agent:
    def __init__(self, state_size):
        """Initialize for Agent."""
        self.state_size = state_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001  # minimum exploration rate
        self.epsilon_end_episode = 2000  # episode to end exploration
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_end_episode

        self.batch_size = 256
        self.replay_start = 3000
        self.epochs = 1

        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'),
            Dense(64, activation='relu', kernel_initializer='glorot_uniform'),
            Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
            Dense(4, activation='linear')  # Assuming 4 possible actions
        ])

        model.compile(loss='mse', optimizer='adam'))
        return model
    
    def add_to_memory(self, state, action, reward, next_state, done):
        """Add experience to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action based on epsilon-greedy policy."""
        max_value = -sys.maxsize - 1
        best = None

        if np.random.rand() <= self.epsilon:
            return np.random.choice([0, 1, 2, 3])  # Random action