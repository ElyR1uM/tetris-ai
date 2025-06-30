"""Trainer for agent.py"""

from tetris_engine import tEngine
from agent import Agent

env = tEngine()
agent = Agent(4)

max_episodes = 3000
max_steps = 25000

episodes = []
rewards = []

for episode in range(max_episodes):
    current_state = env.reset()
    done = False
    steps = 0
    total_reward = 0
    print(f"Training at episode {episode + 1}/{max_episodes}...")

    while not done and steps < max_steps:
        # Run game
        env.step()

        # Perform action
        action = agent.act(state)

        # Fetch all possible placements for the current piece
        next_states = env.get_possible_states()

        # next_states empty means the game is over
        if not next_states:
            break

        # Tells the agent to choose the best action from the possible placements
        best_state = agent.act(next_states.values())

        # Finds the best state and chooses the corresponding action to achieve it
        best_action = None
        for action, state in next_states.items():
            if state == best_state:
                best_action = action
                break

        reward = env.get_reward()
        done = env.game_over
        total_reward += reward

        agent.add_to_memory(current_state, next_states[best_action], reward, done)

        current_state = next_states[best_action]
        
        steps += 1

    print("Total reward: ", str(total_reward))
    episodes.append(episode)
    rewards.append(total_reward)

    if episode % 100 == 0:
        print(f"Episode {episode}/{max_episodes}, Total Reward: {total_reward}")

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon -= agent.epsilon_decay