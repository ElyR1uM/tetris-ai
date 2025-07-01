"""Trainer for agent.py"""

from tetris_engine import tEngine
from agent import Agent
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore


# Config
MODEL_SAVE_PATH = "model/model.h5"
TRAINING_STATE_PATH = "model/training_state.pkl"
SAVE_INTERVAL = 100
PLOT_INTERVAL = 100

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

def save_progress(episodes, rewards, filepath):
    """Saves the training progress to a JSON file."""
    data = {
        "episodes": episodes,
        "rewards": rewards,
        "timestamp": datetime.now().isoformat()
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Progress saved to {filepath}")

def load_progress(filepath):
    """Loads the training progress from a JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data.get("episodes", []), data.get("rewards", [])
    return [], []

def plot_progress(episodes, rewards):
    """Converts the training progress into a graph"""
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label='Total Reward', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig("model/q_graph.png")
    plt.close()
    print(f"Progress plot saved to model/q_graph.png")

env = tEngine()
agent = Agent(4)

max_steps = 50000
max_episodes = 3000

episodes, rewards = load_progress(TRAINING_STATE_PATH)
start_episode = len(episodes)

print(f"Starting training from episode {start_episode + 1}/{max_episodes}...")

for episode in range(max_episodes):
    current_state = env.reset()
    done = False
    steps = 0
    total_reward = 0
    print(f"Training at episode {episode + 1}/{max_episodes}...")

    while not done and steps < max_steps:
        # Run game
        env.step()

        # Fetch all possible placements for the current piece
        next_states = env.get_possible_states()

        # next_states empty means the game is over
        if not next_states:
            break

        # Tells the agent to choose the best action from the possible placements
        best_action = agent.act(next_states)

        if best_action is None:
            print("bext_action returned None")
            break

        next_state = next_states[best_action]

        reward = env.get_reward()
        done = env.game_over
        total_reward += reward

        agent.add_to_memory(current_state, next_state, reward, done)

        current_state = next_state
        
        steps += 1

    print("Total reward: ", str(total_reward))
    episodes.append(episode)
    rewards.append(total_reward)

    if episode % 10 == 0:
        avg_reward = sum(rewards[-10:]) / min(10, len(rewards))
        print(f"Episode {episode + 1}/{max_episodes}, Total Reward: {total_reward}, Avg Last 10: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    if (episode + 1) % SAVE_INTERVAL == 0:
        agent.save_model(MODEL_SAVE_PATH)
        save_progress(episodes, rewards, TRAINING_STATE_PATH)
    
    if (episode + 1) % PLOT_INTERVAL == 0:
        plot_progress(episodes, rewards)

    if episode % 100 == 0:
        print(f"Episode {episode}/{max_episodes}, Total Reward: {total_reward}")

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon -= agent.epsilon_decay