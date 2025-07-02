"""Agent to play tetris_engine.py"""

# For these imports to work you want to have a venv with Python 3.10 (See README)
import tensorflow as tf # type: ignore
tf.config.run_functions_eagerly(True)

import keras # type: ignore
from keras import initializers # type: ignore
from keras.models import Sequential, load_model # type: ignore
from keras.layers import Dense # type: ignore
import numpy as np # type: ignore
from collections import deque
import random
import sys
import pickle
import os

class Agent:
    def __init__(self, state_size, model_path="model/model.h5"):
        self.state_size = state_size
        self.memory = deque(maxlen=50000)
        self.discount = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01 
        self.epsilon_end_episode = 2000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_end_episode

        self.batch_size = 64
        self.replay_start = 1000
        self.epochs = 1
        self.steps = 0

        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = load_model(model_path)
            self.model.compile(loss='huber', optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)) # type: ignore
            target_model_path = model_path.replace('.h5', '_target.h5')
            if os.path.exists(target_model_path):
                self.target_model = load_model(target_model_path)
            else:
                self.target_model = load_model(model_path)  # Fallback to the main model if target model doesn't exist
            
            self.target_model.compile(loss='huber', optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)) # type: ignore
            
            self.load_training_state(model_path.replace('.h5', '_state.pkl'))
        else:
            print("No model found, building a new one.")
            self.model = self.build_model()
            self.target_model = self.build_model()

    def build_model(self):
        model = Sequential([
            keras.Input(shape=(self.state_size,), name='input_layer'),
            Dense(64, activation='leaky_relu', kernel_initializer='he_normal'),
            Dense(64, activation='leaky_relu', kernel_initializer='he_normal'),
            Dense(32, activation='leaky_relu', kernel_initializer='he_normal'),
            Dense(1, activation='linear')
        ])

        model.compile(loss='huber', optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)) # type: ignore
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def add_to_memory(self, current_state, next_state, reward, done):
        self.memory.append((current_state, next_state, reward, done))
    
    def act(self, state_dict):
        """Chooses an action from available actions."""
        if random.random() <= self.epsilon:
            return random.choice(list(state_dict.keys()))
        else:
            max_value = -sys.maxsize - 1
            best_action = None
            for action, s in state_dict.items():
                value = self.model.predict(np.array([s]), verbose=0)[0][0] # type: ignore
                if value > max_value:
                    max_value = value
                    best_action = action

            if best_action is None:
                best_action = random.choice(list(state_dict.keys()))

        return best_action
    
    def save_model(self, filepath):
        """Saves the model and training state"""
        # Save the Network
        self.model.save(filepath) # type: ignore

        target_model_filepath = filepath.replace('.h5', '_target.h5')
        self.target_model.save(target_model_filepath)

        state_filepath = filepath.replace('.h5', '_state.pkl')
        training_state = {
            'epsilon': self.epsilon,
            'memory': list(self.memory),
            'steps': self.steps,
        }
        with open(state_filepath, 'wb') as f:
            pickle.dump(training_state, f)

        print(f"Model saved to {filepath}, target model saved to {target_model_filepath}")
        print(f"Training state saved to {state_filepath}")

    def load_training_state(self, state_filepath):
        """Loads epsilon, steps, and memory from a file"""
        if os.path.exists(state_filepath):
            try:
                with open(state_filepath, 'rb') as f:
                    training_state = pickle.load(f)
                    
                self.epsilon = training_state.get('epsilon', self.epsilon)
                self.steps = training_state.get('steps', 0)
                saved_memory = training_state.get('memory', [])
                self.memory = deque(saved_memory, maxlen=50000)
                print(f"Training state loaded from {state_filepath}")
            except Exception as e:
                print(f"Error loading training state: {e}")
    
    def replay(self):
        if len(self.memory) < self.replay_start:
            return
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory))) # type: ignore

        print(f"Replaying {len(batch)} transitions from memory.")

        states = np.array([transition[0] for transition in batch])
        next_states = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        dones = np.array([transition[3] for transition in batch])

        current_qvalue = self.model.predict(states, verbose=0).numpy() if hasattr(self.model.predict(states, verbose=0), 'numpy') else np.array(self.model.predict(states, verbose=0))

        next_qvalue = self.target_model.predict(next_states, verbose=0)
        if hasattr(next_qvalue, 'numpy'):
            next_qvalue = next_qvalue.numpy()
        else:
            next_qvalue = np.array(next_qvalue)

        target_qvalue = current_qvalue.copy()

        for i in range(len(batch)):
            if dones[i]:
                target_qvalue[i] = rewards[i]
            else:
                target_qvalue[i] = rewards[i] + self.discount * np.amax(next_qvalue[i])

        history = self.model.fit(states, 
                                 target_qvalue, 
                                 batch_size=self.batch_size, 
                                 epochs=self.epochs, 
                                 verbose=0)

        return history.history['loss'][0] if history.history['loss'] else 0