import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from utils.config import Config
from utils.plotters import plot_durations, save_trained_agent_gif
from utils.preprocessing import CarRacingEnv

from agents.RandomAgent import RandomAgent
from agents.QLearningAgent import QLearningAgent
from agents.DQNAgent import DQNAgent, DQN_CNN_Agent
from agents.DDQNAgent import DDQNAgent

# Set up CartPole configuration
cartpole_config = Config()
cartpole_config.environment = gym.make('CartPole-v1', render_mode='rgb_array')
cartpole_config.hyperparameters = {
    'epsilon': 0.9,
    'epsilon_decay': 0.99,
    'gamma': 0.99,
    'alpha': 0.05,
    'num_episodes': 600,
    'max_t': 500,
    'state_range': [(-4.8, 4.8), (-3, 3), (-0.5, 0.5), (-2, 2)],
    'n_state_bins': [7, 7, 7, 7],
    'batch_size': 128,
    'hidden_dims': [128, 128],
    'last_layer_activation': False,
    'learning_rate': 1e-4,
    'tau': 0.005,
    'buffer_size': 10000,
}

# Set up LunarLander configuration
# Not really compatible with Q-learning due to large continuous state space
lunarlander_config = Config()
lunarlander_config.environment = gym.make('LunarLander-v2', render_mode='rgb_array')
lunarlander_config.hyperparameters = {
    'epsilon': 0.9,
    'epsilon_decay': 0.99,
    'gamma': 0.99,
    'num_episodes': 500,
    'max_t': 1000,
    'batch_size': 64,
    'hidden_dims': [64, 64],
    'last_layer_activation': False,
    'learning_rate': 5e-4,
    'tau': 0.001,
    'buffer_size': 10000,
}

# Set up MountainCar configuration
mountaincar_config = Config()
mountaincar_config.environment = gym.make('MountainCar-v0', render_mode='rgb_array')
mountaincar_config.hyperparameters = {
    'epsilon': 0.9,
    'epsilon_decay': 0.99,
    'gamma': 0.99,
    'num_episodes': 1500,
    'max_t': 1000,
    'batch_size': 64,
    'hidden_dims': [64, 64],
    'last_layer_activation': False,
    'learning_rate': 5e-4,
    'tau': 0.001,
    'buffer_size': 10000,
}

# Set up CarRacing configuration
carracing_config = Config()
carracing_config.environment = CarRacingEnv(gym.make('CarRacing-v2', render_mode='rgb_array', continuous=False))
carracing_config.hyperparameters = {
    'epsilon': 1.0,
    'epsilon_decay': 0.99,
    'gamma': 0.99,
    'num_episodes': 600,
    'max_t': 250,
    'batch_size': 32,
    'kernel_sizes': [8, 4],
    'strides': [4, 2],
    'channels': [16, 32],
    'hidden_dims': [256],
    'last_layer_activation': False,
    'learning_rate': 2.5e-4,
    'tau': 0.001,
    'buffer_size': 10000,
    'input_shape': (4, 84, 84),
    'min_epsilon': 0.05,
}

# Example code for training an agent

agent = DQN_CNN_Agent(carracing_config)
episodes, scores = agent.train()
print(agent.PolicyNet)

# Plot episode durations
plot_durations(scores, 
               figsize=(10, 6), 
               moving_average_window=100,
               filename='carracing_dqn'
)

save_trained_agent_gif(agent, 
                       agentname='DQN Agent',
                       filename='carracing_dqn',
                       n_time_steps=250
)