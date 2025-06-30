"""Trainer for agent.py"""

from tetris_engine import tEngine
from agent import Agent

env = tEngine()
agent = Agent(4)

max_episodes = 3000

episodes = []
rewards = []

for episode in range(max_episodes):
    current_state = env.reset()
    done = False
    total_reward = 0
    print(f"Training at episode {episode + 1}/{max_episodes}...")

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step()
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.replay()
    episodes.append(episode)
    rewards.append(total_reward)

    if episode % 100 == 0:
        print(f"Episode {episode}/{max_episodes}, Total Reward: {total_reward}")