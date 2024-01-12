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
from agents.DQNAgent import DQNAgent

class DDQNAgent(DQNAgent):
    def __init__(self, config: Config):
        super().__init__(config)

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

        Q = self.PolicyNet(states).gather(1, actions)
        Q_next = torch.zeros(self.hyperparameters['batch_size'])

        # Double DQN - use PolicyNet to select action and TargetNet to evaluate action
        max_action_indices = self.PolicyNet(non_final_next_states).max(1).indices
        with torch.no_grad():
            Q_next[non_final_mask] = self.TargetNet(non_final_next_states).gather(1, max_action_indices.unsqueeze(1)).squeeze(1)

        Q_exp = (Q_next * self.hyperparameters['gamma']) + rewards
        loss = self.loss(Q, Q_exp.unsqueeze(1))

        return loss