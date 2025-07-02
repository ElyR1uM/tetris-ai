"""Trainer for agent.py"""

from tetris_engine import tEngine
from agent import Agent
from datetime import datetime
import pickle
import json
import os
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore


# Config
MODEL_SAVE_PATH = "model/model.h5"
PROGRESS_PATH = "model/progress.json"
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

episodes, rewards = load_progress(PROGRESS_PATH)
start_episode = len(episodes)

print(f"Starting training from episode {start_episode + 1}/{max_episodes}...")

for episode in range(start_episode, max_episodes):
    env.reset()
    current_state = env.get_state()
    done = False
    steps = 0
    total_reward = 0
    print(f"Training at episode {episode + 1}/{max_episodes}...")

    while not done and steps < max_steps:

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

        rotation, x_pos = best_action

        for _ in range(rotation):
            env.rotate()

        env.piece_x = x_pos

        env.hard_drop()

        next_state = next_states[best_action]
        actual_state = env.get_state()
        if not np.allclose(actual_state, next_state):
            print("Warning: State mismatch!")

        reward = env.get_reward()
        done = env.game_over
        total_reward += reward

        agent.add_to_memory(current_state, next_state, reward, done)

        current_state = next_state

        if episode < agent.epsilon_end_episode:
            agent.epsilon = (episode / agent.epsilon_end_episode) * (1.0 - agent.epsilon_min)
        else:
            agent.epsilon = agent.epsilon_min

        if episode % 50 == 0:
            agent.update_target_model()

        replay_freq = 4 if len(agent.memory) >= 10000 else 10
        if steps % replay_freq == 0 and len(agent.memory) >= agent.replay_start:
            agent.replay()
        
        steps += 1

    print("Total reward: ", str(total_reward))
    episodes.append(episode)
    rewards.append(total_reward)

    if episode % 10 == 0:
        avg_reward = sum(rewards[-10:]) / min(10, len(rewards))
        print(f"Episode {episode + 1}/{max_episodes}, Total Reward: {total_reward}, Avg Last 10: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    if (episode) % SAVE_INTERVAL == 0:
        agent.save_model(MODEL_SAVE_PATH)
        save_progress(episodes, rewards, PROGRESS_PATH)
    
    if (episode + 1) % PLOT_INTERVAL == 0:
        plot_progress(episodes, rewards)

    if episode % 100 == 0:
        print(f"Episode {episode}/{max_episodes}, Total Reward: {total_reward}")