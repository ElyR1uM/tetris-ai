"""Agent to play tetris_engine.py"""

import keras
from keras import initializers
from keras.models import Sequential, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random
import sys
import pickle
import os

class Agent:
    def __init__(self, state_size, model_path=None):
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

        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = load_model(model_path)
            self.load_training_state(model_path.replace('.h5', '_state.pkl'))
        else:
            print("No model found, building a new one.")
            self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
				Dense(64, input_dim=self.state_size, activation='leaky_relu', kernel_initializer='glorot_uniform'),
				Dense(64, activation='leaky_relu', kernel_initializer='glorot_uniform'),
				Dense(32, activation='leaky_relu', kernel_initializer='glorot_uniform'),
				Dense(1, activation='linear')
		])

        model.compile(loss='mse', optimizer='adam')
        return model
    
    def add_to_memory(self, current_state, next_state, reward, done):
        self.memory.append((current_state, next_state, reward, done))
    
    def act(self, state):
        max_value = -sys.maxsize - 1
        best = None

        if random.random() <= self.epsilon:
            return random.choice(list(state))
        else:
            for s in state:
                value = self.model.predict(np.array([s]), verbose=0)[0][0] # type: ignore
                if value > max_value:
                    max_value = value
                    best = s

        return best
    
    def save_model(self, filepath):
        """Saves the model and training state"""
        # Save the Network
        self.model.save(filepath) # type: ignore

        state_filepath = filepath.replace('.h5', '_state.pkl')
        training_state = {
            'epsilon': self.epsilon,
            'memory': list(self.memory)
        }
        with open(state_filepath, 'wb') as f:
            pickle.dump(training_state, f)

        print(f"Model saved to {filepath}")
        print(f"Training state saved to {state_filepath}")

    def load_training_state(self, state_filepath):
        """Loads epsilon, memory from a file"""
        if os.path.exists(state_filepath):
            try:
                with open(state_filepath, 'rb') as f:
                    training_state = pickle.load(f)
                    
                self.epsilon = training_state.get('epsilon', self.epsilon)
                saved_memory = training_state.get('memory', [])
                self.memory = deque(saved_memory, maxlen=30000)
                print(f"Training state loaded from {state_filepath}")
            except Exception as e:
                print(f"Error loading training state: {e}")
    
    def replay(self):
        if len(self.memory) > self.replay_start:
            batch = random.sample(self.memory, self.batch_size)

            next_states = np.array([s[1] for s in batch])
            next_qvalue = np.array([s[0] for s in self.model.predict(next_states)]) # type: ignore

            x = []
            y = []


            for i in range(self.batch_size):
                state, _, reward, done = batch[i][0], None, batch[i][2], batch[i][3]
                if not done:
                    new_q = reward + self.discount * next_qvalue[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)

            self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, epochs=self.epochs, verbose=0) # type: ignore