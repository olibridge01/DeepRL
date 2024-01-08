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

from utils import Config, ReplayBuffer, plot_durations
from networks import MLP


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
        self.num_states = (self.hyperparameters['n_state_bins'] + 1) ** len(self.hyperparameters['state_range'])
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
        discrete_state = sum([x * ((self.hyperparameters['n_state_bins'] + 1) ** i) 
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
    
class DQNAgent(object):
    def __init__(self, config : Config):
        self.env = config.environment
        self.hyperparameters = config.hyperparameters
        self.n_actions = self.env.action_space.n
        self.n_obs = self.env.observation_space.shape[0]
        self.memory = ReplayBuffer(self.hyperparameters['buffer_size'])

        self.PolicyNet = MLP(
            input_dims=self.n_obs,
            output_dims=self.n_actions,
            hidden_dims=self.hyperparameters['hidden_dims'],
            activation=nn.ReLU(),
            activation_last_layer=self.hyperparameters['last_layer_activation']
        )
        self.TargetNet = MLP(
            input_dims=self.n_obs,
            output_dims=self.n_actions,
            hidden_dims=self.hyperparameters['hidden_dims'],
            activation=nn.ReLU(),
            activation_last_layer=self.hyperparameters['last_layer_activation']
        )
        self.TargetNet.load_state_dict(self.PolicyNet.state_dict())

        self.optimizer = optim.Adam(self.PolicyNet.parameters(), lr=self.hyperparameters['learning_rate'])
        self.loss = F.smooth_l1_loss

    def get_action(self, state: torch.Tensor) -> int:
        """Returns action based on epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return torch.tensor([[np.random.randint(self.n_actions)]], dtype=torch.long)
        else:
            with torch.no_grad():
                return self.PolicyNet(state).max(1).indices.view(1, 1)
            
    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """Computes loss for a batch of transitions."""
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        Q_next = torch.zeros(self.hyperparameters['batch_size'])
        with torch.no_grad():   
            Q_next[non_final_mask] = self.TargetNet(non_final_next_states).max(1).values
        Q = self.PolicyNet(states).gather(1, actions)
        Q_exp = rewards + self.hyperparameters['gamma'] * Q_next

        # Compute loss
        loss = self.loss(Q, Q_exp.unsqueeze(1))

        return loss
            
    def update(self) -> None:
        """Takes an optimization step."""
        if len(self.memory) < self.hyperparameters['batch_size']:
            return

        # Sample from memory and compute loss
        batch = self.memory.sample(self.hyperparameters['batch_size'])
        loss = self.compute_loss(batch)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.PolicyNet.parameters(), 100)
        self.optimizer.step()
    
    def train(self) -> list:
        """Train DQN agent."""
        # Get necessary hyperparameters
        self.epsilon = self.hyperparameters['epsilon']
        self.epsilon_decay = self.hyperparameters['epsilon_decay']
        num_episodes = self.hyperparameters['num_episodes']
        max_t = self.hyperparameters['max_t']
        tau = self.hyperparameters['tau']

        # Keep track of episode durations
        episode_durations = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            self.epsilon *= self.epsilon_decay

            for t in count():
                # Select action
                action = self.get_action(state)

                # Take action and observe next state
                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward])
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                # Add transition to replay buffer
                self.memory.add_transition(state, action, next_state, reward)

                # Update state and action
                state = next_state

                # Update policy
                self.update()

                # Update target network
                target_net_state_dict = self.TargetNet.state_dict()
                policy_net_state_dict = self.PolicyNet.state_dict()

                for name in target_net_state_dict:
                    target_net_state_dict[name] = tau * policy_net_state_dict[name] + (1 - tau) * target_net_state_dict[name]

                self.TargetNet.load_state_dict(target_net_state_dict)

                if done:
                    print(f'Episode {episode} finished after {t + 1} timesteps')
                    episode_durations.append(t + 1)
                    break

        return episode_durations
    
    
if __name__ == '__main__':

    # Set up CartPole configuration
    cartpole_config = Config()
    cartpole_config.environment = gym.make('CartPole-v1')
    cartpole_config.hyperparameters = {
        'epsilon': 0.9,
        'epsilon_decay': 0.99,
        'gamma': 0.99,
        'alpha': 0.05,
        'num_episodes': 500,
        'max_t': 500,
        'state_range': [(-4.8, 4.8), (-3, 3), (-0.5, 0.5), (-2, 2)],
        'n_state_bins': 7,
        'batch_size': 128,
        'hidden_dims': [128, 64],
        'last_layer_activation': False,
        'learning_rate': 1e-4,
        'tau': 0.005,
        'buffer_size': 10000,
    }

    # Define and train Q-Learning agent
    agent = QLearningAgent(cartpole_config)
    episode_durations = agent.train()

    # Plot episode durations
    plot_durations(episode_durations, 
                   figsize=(10, 6), 
                   moving_average_window=100
    )

    # Define and train DQN agent
    agent = DQNAgent(cartpole_config)
    episode_durations = agent.train()

    # Plot episode durations
    plot_durations(episode_durations, 
                   figsize=(10, 6), 
                   moving_average_window=100
    )


 

