import gymnasium as gym
import numpy as np
import random

from utils.config import Config

class QLearningAgent(object):
    """Agent that learns optimal policy using Q-learning by discretising the state space."""
    def __init__(self, config : Config):
        self.env = config.environment
        self.hyperparameters = config.hyperparameters
        self.num_states = np.prod(self.hyperparameters['n_state_bins'])
        self.Q = np.zeros((self.num_states, self.env.action_space.n))

    def get_discrete_state(self, obs: np.ndarray) -> int:
        """Converts continuous state into discrete state."""
        discrete_state = []
        for i, state_variable in enumerate(obs):
            state_bounds = np.linspace(self.hyperparameters['state_range'][i][0], 
                                       self.hyperparameters['state_range'][i][1], 
                                       self.hyperparameters['n_state_bins'][i] + 1)[1:-1]
            discrete_state.append(np.digitize(state_variable, state_bounds) - 1)

        # Concatenate into single integer 
        discrete_state = sum([x * (self.hyperparameters['n_state_bins'][i] ** i) 
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
            score = 0
            
            for t in range(max_t):
                # Take action and observe next state
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                score += reward

                # Penalise agent if it terminates before max_t
                if terminated and t < max_t - 1:
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
                    print(f'Episode {episode:04d} | Steps: {(t+1):04d} | Score: {score:.2f}')
                    episode_durations.append(t + 1)
                    break
            
        return episode_durations