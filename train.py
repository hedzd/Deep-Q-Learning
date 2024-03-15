import gym
import numpy as np

env=gym.make('CartPole-v1')
NUM_EPISODES = 10
MAX_TIMESTEPS = 1000

# The main program loop
for i_episode in range(NUM_EPISODES):
    observation = env.reset()
    # Iterating through time steps within an episode
    for t in range(MAX_TIMESTEPS):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            # If the pole has tipped over, end this episode
            break