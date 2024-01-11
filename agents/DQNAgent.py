import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.plotters import plot_durations, save_trained_agent_gif
from utils.config import Config
from utils.experience_replay import ReplayBuffer
from networks.MLP import MLP

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
            score = 0
            for t in count():
                # Select action
                action = self.get_action(state)

                # Take action and observe next state
                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                score += reward
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

                # Soft update target network
                target_net_state_dict = self.TargetNet.state_dict()
                policy_net_state_dict = self.PolicyNet.state_dict()

                for name in target_net_state_dict:
                    target_net_state_dict[name] = tau * policy_net_state_dict[name] + (1 - tau) * target_net_state_dict[name]

                self.TargetNet.load_state_dict(target_net_state_dict)

                if done:
                    print(f'Episode {episode:04d} | Steps: {(t+1):04d} | Score: {score:.2f}')
                    episode_durations.append(t + 1)
                    break

        return episode_durations