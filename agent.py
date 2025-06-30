"""Agent to play tetris_engine.py"""

import keras
from keras import initializers
from keras.models import Sequential, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random
import sys

class Agent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.memory = deque(maxlen=30000)
        self.discount = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001 
        self.epsilon_end_episode = 2000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_end_episode

        self.batch_size = 512
        self.replay_start = 3000
        self.epochs = 1

        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
				Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'),
				Dense(64, activation='relu', kernel_initializer='glorot_uniform'),
				Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
				Dense(1, activation='linear')
		])

        model.compile(loss='mse', optimizer='adam')
        return model