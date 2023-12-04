import gymnasium as gym
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
# from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import Config, plot_durations


class RandomAgent(object):
    """Agent that acts randomly."""
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def act(self) -> int:
        """Randomly sample from action space."""
        return self.env.action_space.sample()


class QLearningAgent(object):
    """Agent that learns optimal policy using Q-learning."""
    def __init__(self, config : Config):
        self.env = config.environment
        self.hyperparameters = config.hyperparameters
        self.num_states = (self.hyperparameters['n_state_bins']+1) ** len(self.hyperparameters['state_range'])
        self.Q = np.zeros((self.num_states, self.env.action_space.n))

    def get_discrete_state(self, obs: np.ndarray) -> int:
        """Converts continuous state into discrete state."""
        discrete_state = []
        for i, state_variable in enumerate(obs):
            state_bounds = np.linspace(self.hyperparameters['state_range'][i][0], 
                                       self.hyperparameters['state_range'][i][1], 
                                       self.hyperparameters['n_state_bins'] + 1)[1:-1]
            discrete_state.append(np.digitize(state_variable, state_bounds) - 1)

        # Concatenate into single integer 
        discrete_state = sum([x * ((self.hyperparameters['n_state_bins']+1) ** i) 
                              for i, x in enumerate(discrete_state)])
        
        return discrete_state
    
    def get_max_Q(self, state: int) -> float:
        """Returns maximum Q value for given state."""
        return np.amax(self.Q[state])
    
    def get_action(self, state: int) -> int:
        """Returns action based on epsilon-greedy policy."""

        # Select random action with probability epsilon
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # Select greedy action
            max_Q = self.get_max_Q(state)
            possible_actions = np.argwhere(self.Q[state] == max_Q).flatten()
            return random.choice(possible_actions)
        
    def update_Q(self, state: int, action: int, reward: int, next_state: int) -> None:
        """Updates Q table using Q-learning update rule."""
        alpha = self.hyperparameters['alpha']
        gamma = self.hyperparameters['gamma']
        max_Q = self.get_max_Q(next_state)
        self.Q[state, action] += alpha * (reward + gamma * max_Q - self.Q[state, action])
        return
    
    def train(self) -> list:
        """Runs Q-learning algorithm."""
        # Get necessary hyperparameters
        self.epsilon = self.hyperparameters['epsilon']
        self.epsilon_decay = self.hyperparameters['epsilon_decay']
        num_episodes = self.hyperparameters['num_episodes']
        max_t = self.hyperparameters['max_t']

        # Keep track of episode durations
        episode_durations = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            state = self.get_discrete_state(obs)
            action = self.get_action(state)
            self.epsilon *= self.epsilon_decay # Decay epsilon

            # Reset bool variables
            terminated = False
            truncated = False
            done = False
            
            for timestep in range(max_t):
                # Take action and observe next state
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Penalise agent if it terminates before max_t
                if terminated and timestep < max_t - 1:
                    reward = -num_episodes

                # Discretise next state and get next action
                next_state = self.get_discrete_state(next_obs)
                next_action = self.get_action(next_state)  

                # Update Q table              
                self.update_Q(state, action, reward, next_state)

                # Update state and action
                state = next_state
                action = next_action

                if done:
                    print(f'Episode {episode} finished after {timestep + 1} timesteps')
                    episode_durations.append(timestep + 1)
                    break
            
        return episode_durations


if __name__ == '__main__':

    # Set up CartPole configuration
    cartpole_config = Config()
    cartpole_config.environment = gym.make('CartPole-v1')
    cartpole_config.hyperparameters = {
        'epsilon': 0.5,
        'epsilon_decay': 0.99,
        'gamma': 0.95,
        'alpha': 0.05,
        'num_episodes': 1000,
        'max_t': 500,
        'state_range': [(-4.8, 4.8), (-3, 3), (-0.5, 0.5), (-2, 2)],
        'n_state_bins': 7
    }

    # Define and train agent
    agent = QLearningAgent(cartpole_config)
    episode_durations = agent.train()

    # Plot episode durations
    plot_durations(episode_durations, 
                   figsize=(10, 6), 
                   moving_average_window=100
    )

 

